"""QS-SAM3D 3D segmentation."""

from __future__ import annotations

import copy
import logging
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pointops
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, build_sam
from tqdm.auto import tqdm

from config import QsSam3dConfig
from utils.gpu import gpu_cleanup
from utils.spatial import (
    Voxelize,
    get_matching_indices,
    make_open3d_pcd,
    num_to_natural,
    pairwise_indices,
    remove_small_group,
)
from utils.visualizer import (
    visualize_masks_2d,
    visualize_mesh_partition,
    visualize_mesh_segmentation,
    visualize_partition,
)

logger = logging.getLogger(__name__)


class QsSam3dSegmentation:
    """SAM-based 3D segmentation with hierarchical point-cloud fusion."""

    def __init__(self, config: QsSam3dConfig) -> None:
        self.config = config
        if not Path(config.sam_model_path).is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {config.sam_model_path}")

        logger.info("Loading SAM ViT-H from %s", config.sam_model_path)
        t0 = time.perf_counter()
        with gpu_cleanup():
            sam = build_sam(checkpoint=str(config.sam_model_path)).to("cuda")
            self.mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=config.points_per_side,
                pred_iou_thresh=config.pred_iou_thresh,
                stability_score_thresh=config.stability_score_thresh,
                crop_n_layers=config.crop_n_layers,
                crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
                min_mask_region_area=config.min_mask_region_area,
            )
        self.voxelize = Voxelize(voxel_size=config.voxel_size, keys=("coord", "color", "group"))
        logger.info("SAM loaded in %.1fs", time.perf_counter() - t0)

    def run(self) -> None:
        cfg = self.config

        if not cfg.sam3d_dir.is_dir():
            raise FileNotFoundError(f"Scene dir missing: {cfg.sam3d_dir}")
        if not cfg.sam3d_pointcloud.is_file():
            raise FileNotFoundError(f"Scene pointcloud missing: {cfg.sam3d_pointcloud}")

        if cfg.sam3d_seg_dir.exists():
            shutil.rmtree(cfg.sam3d_seg_dir)
        cfg.sam3d_seg_dir.mkdir(parents=True)

        t_total = time.perf_counter()

        logger.info("Step 1/4: Load frames + SAM + back-project")
        t0 = time.perf_counter()
        pcd_list = self._load_scene_frames()
        if not pcd_list:
            raise RuntimeError("No valid per-frame point clouds produced")
        logger.info("Step 1 done: %d frames in %.1fs", len(pcd_list), time.perf_counter() - t0)

        if len(pcd_list) >= cfg.quality_window_size:
            logger.info("Step 2/4: Quality filtering")
            t0 = time.perf_counter()
            with gpu_cleanup():
                pcd_list = self._filter_frames(pcd_list)
            if not pcd_list:
                raise RuntimeError("All frames rejected by quality filter")
            logger.info("Step 2 done: %d kept in %.1fs", len(pcd_list), time.perf_counter() - t0)
        else:
            logger.info("Step 2/4: skipped (too few frames)")

        logger.info("Step 3/4: Hierarchical merge")
        t0 = time.perf_counter()
        with gpu_cleanup():
            merged = self._merge_clouds(pcd_list)
        logger.info("Step 3 done in %.1fs", time.perf_counter() - t0)

        logger.info("Step 4/4: Finalize")
        t0 = time.perf_counter()
        with gpu_cleanup():
            self._finalize(merged)
        logger.info("Step 4 done in %.1fs", time.perf_counter() - t0)

        logger.info("Pipeline done in %.1fs total", time.perf_counter() - t_total)

    def _load_scene_frames(self) -> list[dict]:
        cfg = self.config
        color_files = sorted(cfg.sam3d_color_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        total = len(color_files)
        logger.info("Found %d color frames", total)

        cfg.sam3d_masks_dir.mkdir(parents=True, exist_ok=True)
        pcd_list: list[dict] = []
        bar = tqdm(color_files, desc="SAM3D frames", unit="frame")
        for color_path in bar:
            idx = int(color_path.stem)
            pcd = self._frame_to_pcd(color_path, idx)
            if pcd is None:
                logger.warning("Frame idx=%d skipped (no valid depth)", idx)
                continue
            with gpu_cleanup():
                pcd = self.voxelize(pcd)
            pcd_list.append(pcd)
            bar.set_postfix(idx=idx, pts=pcd["coord"].shape[0])
        return pcd_list

    def _frame_to_pcd(self, color_path: Path, idx: int) -> dict | None:
        cfg = self.config

        color_bgr = cv2.imread(str(color_path))
        depth_orig = cv2.imread(str(cfg.sam3d_depth_dir / f"{idx}.png"), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_orig is None:
            return None

        tw, th = cfg.image_size
        color = cv2.resize(color_bgr, (tw, th))
        depth = cv2.resize(depth_orig, (tw, th), interpolation=cv2.INTER_NEAREST)

        group_ids_full = self._run_sam(color)
        Image.fromarray(num_to_natural(group_ids_full).astype(np.int16), mode="I;16").save(
            cfg.sam3d_masks_dir / f"{color_path.stem}.png"
        )

        mask_valid = depth != 0
        colors = color[mask_valid][:, ::-1].copy()  # OpenCV BGR → RGB
        group_ids = group_ids_full[mask_valid]

        intr_path = cfg.sam3d_intrinsics_dir / f"{idx}.txt"
        if not intr_path.is_file():
            intr_path = cfg.sam3d_intrinsic_global
        intr = np.loadtxt(intr_path)

        scale_x = tw / color_bgr.shape[1]
        scale_y = th / color_bgr.shape[0]
        fx, fy = intr[0, 0] * scale_x, intr[1, 1] * scale_y
        cx, cy = intr[0, 2] * scale_x, intr[1, 2] * scale_y
        bx, by = intr[0, 3] * scale_x, intr[1, 3] * scale_y

        h, w = depth.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        uv = np.stack([xs, ys, depth / cfg.depth_shift], axis=-1).reshape(-1, 3)
        uv = uv[uv[:, 2] != 0]
        if uv.shape[0] == 0:
            return None

        pts = np.ones((uv.shape[0], 4), dtype=np.float64)
        pts[:, 0] = (uv[:, 0] - cx) * uv[:, 2] / fx + bx
        pts[:, 1] = (uv[:, 1] - cy) * uv[:, 2] / fy + by
        pts[:, 2] = uv[:, 2]
        pose = np.loadtxt(cfg.sam3d_pose_dir / f"{idx}.txt")
        world = pts @ pose.T

        return dict(coord=world[:, :3], color=colors, group=num_to_natural(group_ids))

    def _run_sam(self, image_bgr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            masks = self.mask_generator.generate(image_bgr)
        group_ids = np.full(image_bgr.shape[:2], -1, dtype=np.int64)
        for i in reversed(range(len(masks))):
            group_ids[masks[i]["segmentation"]] = len(masks) - 1 - i
        return group_ids

    def _filter_frames(self, pcd_list: list[dict]) -> list[dict]:
        cfg = self.config
        n = len(pcd_list)
        margin = cfg.quality_window_size // 2
        search = cfg.match_distance_factor * cfg.voxel_size

        # Cache Open3D wrappers once per frame — each frame is scored with up to
        # `quality_window_size` neighbors, so converting on every call wastes work.
        o3d_pcds = [make_open3d_pcd(pcd["coord"]) for pcd in pcd_list]

        # Drop the first/last `margin` frames: they lack a full neighbor window,
        # and capture sequences typically start/end on weakly-covered viewpoints
        # whose unscored inclusion would pollute downstream merging.
        if margin > 0:
            logger.info("Dropping %d boundary frames (margin=%d) before scoring", 2 * margin, margin)

        scores: dict[int, float] = {}
        for i in tqdm(range(margin, n - margin), desc="Quality scoring", unit="frame"):
            window = []
            for j in range(i - margin, i + margin + 1):
                if j == i or j < 0 or j >= n:
                    continue
                s_ij = _match_score(pcd_list[i], o3d_pcds[i], pcd_list[j], o3d_pcds[j], search)
                s_ji = _match_score(pcd_list[j], o3d_pcds[j], pcd_list[i], o3d_pcds[i], search)
                window.append((s_ij + s_ji) / 2.0)
            scores[i] = statistics.median(window) if window else 0.0

        if len(scores) < 3:
            logger.info("Too few scored frames; keeping all")
            return pcd_list

        threshold = float(np.percentile(list(scores.values()), cfg.frame_rejection_percent))
        kept = sorted(i for i, s in scores.items() if s >= threshold)
        logger.info(
            "Quality threshold=%.4f, kept %d/%d scored frames",
            threshold, len(kept), len(scores),
        )
        return [pcd_list[i] for i in kept]

    def _merge_clouds(self, pcd_list: list[dict]) -> dict:
        cfg = self.config
        rnd = 0
        while len(pcd_list) > 1:
            rnd += 1
            pairs = pairwise_indices(len(pcd_list))
            t0 = time.perf_counter()
            new_list: list[dict] = []
            for indices in pairs:
                with gpu_cleanup():
                    merged = _merge_pair(
                        pcd_list, indices, cfg.voxel_size, self.voxelize,
                        cfg.match_distance_factor, cfg.overlap_ratio,
                    )
                if merged is not None:
                    new_list.append(merged)
            pcd_list = new_list
            logger.info(
                "Merge round %d: %d pairs -> %d clouds (%.1fs)",
                rnd, len(pairs), len(pcd_list), time.perf_counter() - t0,
            )
            if not pcd_list:
                raise RuntimeError(f"Merge round {rnd} produced no clouds")
        return pcd_list[0]

    def _finalize(self, seg_dict: dict) -> None:
        cfg = self.config
        seg_dict["group"] = num_to_natural(
            remove_small_group(seg_dict["group"], cfg.min_group_size)
        )
        n_groups = len(np.unique(seg_dict["group"][seg_dict["group"] >= 0]))
        logger.info("Merged cloud: %d points, %d groups", seg_dict["coord"].shape[0], n_groups)

        visualize_partition(
            seg_dict["coord"], seg_dict["group"], cfg.sam3d_merged_cloud, cfg.viz.color_seed
        )

        data = torch.load(cfg.sam3d_pointcloud, weights_only=False)
        with torch.no_grad():
            scene = torch.as_tensor(data["coord"]).cuda().contiguous().float()
            gen = torch.as_tensor(seg_dict["coord"]).cuda().contiguous().float()
            scene_off = torch.tensor([scene.shape[0]], device="cuda", dtype=torch.int32)
            gen_off = torch.tensor([gen.shape[0]], device="cuda", dtype=torch.int32)
            idx, dis = pointops.knn_query(1, gen, gen_off, scene, scene_off)  # type: ignore[misc]

        idx_np = idx.cpu().numpy().reshape(-1)
        dis_np = dis.cpu().numpy().reshape(-1)
        labels = seg_dict["group"][idx_np].astype(np.int16)
        far = dis_np > cfg.distance_threshold
        labels[far] = -1

        # A drifted frame seeds whole runs of points as small noise clusters over
        # the dense scene; a uniform K-neighbour majority vote erodes them, peeling
        # one layer off each cluster's rim per pass. Neighbourhoods never change,
        # so KNN is computed once and reused across iterations. The pairwise-eq
        # tensor is chunked over N, pinning peak memory to the chunk rather than
        # to the total point count.
        del scene, gen, scene_off, gen_off, idx, dis  # free the 1-NN stage's GPU tensors
        with torch.no_grad():
            pts = torch.as_tensor(data["coord"]).cuda().contiguous().float()
            off = torch.tensor([pts.shape[0]], device="cuda", dtype=torch.int32)
            knn_idx, _ = pointops.knn_query(cfg.smooth_k, pts, off, pts, off)  # type: ignore[misc]
            cur = torch.as_tensor(labels, device="cuda", dtype=torch.long)
            chunk = 200_000
            for _ in range(cfg.smooth_iters):
                nxt = torch.empty_like(cur)
                for s in range(0, cur.shape[0], chunk):
                    e = min(s + chunk, cur.shape[0])
                    nbr = cur[knn_idx[s:e]]                                                # (B, K)
                    best = (nbr.unsqueeze(2) == nbr.unsqueeze(1)).sum(dim=1).argmax(dim=1, keepdim=True)
                    nxt[s:e] = nbr.gather(1, best).squeeze(1)
                cur = nxt
            labels = cur.cpu().numpy().astype(np.int16)

        labels = num_to_natural(labels).astype(np.int16)
        logger.info(
            "Background %d/%d (distance > %.3f)", int(far.sum()), far.size, cfg.distance_threshold
        )

        torch.save(labels, cfg.sam3d_segmentation)
        visualize_partition(
            np.asarray(data["coord"]), labels, cfg.sam3d_segmentation_ply, cfg.viz.color_seed
        )
        logger.info("Segmentation → %s", cfg.sam3d_segmentation)

        if not cfg.openmvs_mesh_ply.is_file():
            raise FileNotFoundError(f"OpenMVS mesh missing: {cfg.openmvs_mesh_ply}")
        mesh = o3d.io.read_triangle_mesh(str(cfg.openmvs_mesh_ply))
        verts = np.asarray(mesh.vertices)
        # After the loop nxt and cur alias one tensor; both names must go before
        # the memory held since the KNN adjacency table is actually released.
        del pts, off, knn_idx, cur
        if cfg.smooth_iters:
            del nxt
        with torch.no_grad():
            src = torch.as_tensor(data["coord"]).cuda().contiguous().float()
            tgt = torch.from_numpy(verts).cuda().contiguous().float()
            src_off = torch.tensor([src.shape[0]], device="cuda", dtype=torch.int32)
            tgt_off = torch.tensor([tgt.shape[0]], device="cuda", dtype=torch.int32)
            m_idx, m_dis = pointops.knn_query(1, src, src_off, tgt, tgt_off)  # type: ignore[misc]
        m_idx_np = m_idx.cpu().numpy().reshape(-1)
        m_dis_np = m_dis.cpu().numpy().reshape(-1)
        mesh_labels = labels[m_idx_np].copy()
        mesh_far = m_dis_np > cfg.distance_threshold
        mesh_labels[mesh_far] = -1
        logger.info(
            "Mesh mapping: %d/%d verts labeled (distance ≤ %.3f)",
            int((~mesh_far).sum()), mesh_labels.size, cfg.distance_threshold,
        )
        visualize_mesh_partition(
            mesh, mesh_labels, cfg.sam3d_mesh_segmentation_ply, cfg.viz.color_seed
        )
        logger.info("Mesh segmentation → %s", cfg.sam3d_mesh_segmentation_ply)

        visualize_mesh_segmentation(
            mesh, mesh_labels, cfg.sam3d_segmented_mesh_ply,
            cfg.viz.alpha, cfg.viz.beta, cfg.viz.color_seed,
        )
        visualize_masks_2d(
            mask_dir=cfg.sam3d_masks_dir,
            rgb_dir=cfg.sam3d_color_dir,
            output_dir=cfg.sam3d_viz_dir,
            alpha=cfg.viz.alpha,
            beta=cfg.viz.beta,
            seed=cfg.viz.color_seed,
        )
        logger.info("Visualizations → %s", cfg.sam3d_viz_dir)


def _match_score(
    pcd_a: dict,
    o3d_a: o3d.geometry.PointCloud | None,
    pcd_b: dict,
    o3d_b: o3d.geometry.PointCloud | None,
    search_radius: float,
) -> float:
    """Bidirectional best-match IoU between two per-frame clouds."""
    if o3d_a is None or o3d_b is None:
        return 0.0

    matches = get_matching_indices(o3d_a, o3d_b, search_radius)
    group_a, group_b = pcd_a["group"], pcd_b["group"]

    counts_a = dict(zip(*np.unique(group_a, return_counts=True)))
    counts_b = dict(zip(*np.unique(group_b, return_counts=True)))

    overlap: dict[int, dict[int, int]] = {}
    for i, j in matches:
        ga, gb = int(group_a[i]), int(group_b[j])
        if ga == -1 or gb == -1:
            continue
        overlap.setdefault(ga, {})
        overlap[ga][gb] = overlap[ga].get(gb, 0) + 1

    ious: list[float] = []
    for ga, ov in overlap.items():
        gb = max(ov, key=ov.get)  # type: ignore[arg-type]
        count = ov[gb]
        union = float(counts_a.get(ga, 0) + counts_b.get(gb, 0) - count)
        if union > 0:
            ious.append(count / union)
    return statistics.mean(ious) if ious else 0.0


def _cal_group(
    base_dict: dict, new_dict: dict, match_inds: list[list[int]], ratio: float
) -> np.ndarray:
    """Remap `new_dict["group"]` into `base_dict["group"]`'s ID space via IoU."""
    group_base, group_new = base_dict["group"], new_dict["group"]
    if group_base.size == 0 or group_new.size == 0:
        return group_new

    group_new[group_new != -1] += int(group_base.max()) + 1

    counts_base = dict(zip(*np.unique(group_base, return_counts=True)))
    counts_new = dict(zip(*np.unique(group_new, return_counts=True)))

    overlap: dict[int, dict[int, int]] = {}
    for i, j in match_inds:
        gi, gj = int(group_new[i]), int(group_base[j])
        if gi == -1:
            group_new[i] = group_base[j]
            continue
        if gj == -1:
            continue
        overlap.setdefault(gi, {})
        overlap[gi][gj] = overlap[gi].get(gj, 0) + 1

    for gi, ov in overlap.items():
        gj = max(ov, key=ov.get)  # type: ignore[arg-type]
        count = ov[gj]
        if gj not in counts_base or gi not in counts_new:
            continue
        union = float(counts_base[gj] + counts_new[gi] - count)
        if union > 0 and count / union >= ratio:
            group_new[group_new == gi] = gj
    return group_new


def _merge_pair(
    pcd_list: list[dict],
    indices: list[int],
    voxel_size: float,
    voxelize: Voxelize,
    match_distance_factor: float,
    overlap_ratio: float,
) -> dict | None:
    if len(indices) == 1:
        return pcd_list[indices[0]]

    d0, d1 = pcd_list[indices[0]], pcd_list[indices[1]]
    pcd0 = make_open3d_pcd(d0["coord"])
    pcd1 = make_open3d_pcd(d1["coord"])
    if pcd0 is None:
        return d1 if pcd1 is not None else None
    if pcd1 is None:
        return d0

    r = match_distance_factor * voxel_size
    matches_10 = get_matching_indices(pcd1, pcd0, r)
    d1["group"] = _cal_group(d0, copy.deepcopy(d1), matches_10, overlap_ratio)
    matches_01 = get_matching_indices(pcd0, pcd1, r)
    new_g0 = _cal_group(d1, copy.deepcopy(d0), matches_01, overlap_ratio)

    merged = dict(
        coord=np.concatenate([d0["coord"], d1["coord"]]),
        color=np.concatenate([d0["color"], d1["color"]]),
        group=num_to_natural(np.concatenate([new_g0, d1["group"]])),
    )
    return voxelize(merged)


if __name__ == "__main__":
    from utils.cli import run_stage

    run_stage(__file__, "QS-SAM3D 3D segmentation", QsSam3dConfig.from_yaml, QsSam3dSegmentation)
