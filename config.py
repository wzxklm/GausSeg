"""Configuration system: YAML -> dataclass."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module configs
# ---------------------------------------------------------------------------

@dataclass
class ColmapConfig:
    """COLMAP SfM configuration."""

    project_dir: Path
    extractor: str
    matcher: str
    max_num_features: int
    resize_max: int
    resize_force: bool
    nms_radius: int
    pair_strategy: str
    vocab_tree_path: str
    num_images: int
    num_nearest_neighbors: int
    retrieval_model: str
    num_matched: int
    lightglue_depth_confidence: float
    lightglue_width_confidence: float
    mapper: str
    glomap_refine: bool
    camera_mode: str
    camera_model: str

    # Derived paths (computed in __post_init__)
    input_images: Path = field(init=False)
    output_dir: Path = field(init=False)
    colmap_dir: Path = field(init=False)
    colmap_database: Path = field(init=False)
    colmap_sparse_parent: Path = field(init=False)
    colmap_sparse: Path = field(init=False)
    colmap_dense: Path = field(init=False)
    colmap_images: Path = field(init=False)
    hloc_dir: Path = field(init=False)
    hloc_pairs: Path = field(init=False)
    hloc_features: Path = field(init=False)
    hloc_matches: Path = field(init=False)
    hloc_global_features: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.input_images = self.project_dir / "inputs" / "images"
        self.output_dir = self.project_dir / "output"
        self.colmap_dir = self.output_dir / "colmap"
        self.colmap_database = self.colmap_dir / "database.db"
        self.colmap_sparse_parent = self.colmap_dir / "sparse"
        self.colmap_sparse = self.colmap_sparse_parent / "0"
        self.colmap_dense = self.colmap_dir / "dense"
        self.colmap_images = self.colmap_dense / "images"
        self.hloc_dir = self.colmap_dir / "hloc"
        self.hloc_pairs = self.hloc_dir / "pairs.txt"
        self.hloc_features = self.hloc_dir / "features.h5"
        self.hloc_matches = self.hloc_dir / "matches.h5"
        self.hloc_global_features = self.hloc_dir / "global_features.h5"

        import pycolmap
        valid_extractors = {"sift", "aliked_n16rot", "aliked_n32", "superpoint"}
        valid_matchers = {
            n.split("_", 1)[1].lower()
            for n in pycolmap.FeatureMatcherType.__members__
            if n.startswith(("SIFT_", "ALIKED_"))
        }
        if self.extractor not in valid_extractors:
            raise ValueError(
                f"Invalid extractor {self.extractor!r}. Valid: {sorted(valid_extractors)}"
            )
        if self.matcher not in valid_matchers:
            raise ValueError(
                f"Invalid matcher {self.matcher!r}. Valid: {sorted(valid_matchers)}"
            )
        if self.max_num_features <= 0:
            raise ValueError(f"max_num_features must be > 0, got {self.max_num_features}")
        if self.resize_max <= 0:
            raise ValueError(f"resize_max must be > 0, got {self.resize_max}")
        if self.nms_radius <= 0:
            raise ValueError(f"nms_radius must be > 0, got {self.nms_radius}")

        valid_pair_strategies = {"exhaustive", "vocab_tree", "retrieval"}
        if self.pair_strategy not in valid_pair_strategies:
            raise ValueError(
                f"Invalid pair_strategy {self.pair_strategy!r}. "
                f"Valid: {sorted(valid_pair_strategies)}"
            )
        if self.pair_strategy == "vocab_tree":
            # pycolmap's vocab tree was trained on SIFT descriptors; feeding ALIKED
            # descriptors through match_vocabtree produces garbage matches.
            if self.extractor != "sift":
                raise ValueError(
                    f"pair_strategy=vocab_tree requires extractor=sift, "
                    f"got extractor={self.extractor!r}"
                )
            if not self.vocab_tree_path:
                raise ValueError("pair_strategy=vocab_tree requires vocab_tree_path")
            if not Path(self.vocab_tree_path).exists():
                raise FileNotFoundError(f"Vocab tree not found: {self.vocab_tree_path}")
        if self.pair_strategy == "retrieval":
            # pairs_from_retrieval lives in the hloc branch, which is only
            # wired up for SuperPoint + LightGlue.
            if self.extractor != "superpoint":
                raise ValueError(
                    f"pair_strategy=retrieval requires extractor=superpoint, "
                    f"got extractor={self.extractor!r}"
                )
            valid_retrieval_models = {"netvlad", "megaloc"}
            if self.retrieval_model not in valid_retrieval_models:
                raise ValueError(
                    f"Invalid retrieval_model {self.retrieval_model!r}. "
                    f"Valid: {sorted(valid_retrieval_models)}"
                )
        if self.num_images <= 0:
            raise ValueError(f"num_images must be > 0, got {self.num_images}")
        if self.num_nearest_neighbors <= 0:
            raise ValueError(
                f"num_nearest_neighbors must be > 0, got {self.num_nearest_neighbors}"
            )
        if self.num_matched <= 0:
            raise ValueError(f"num_matched must be > 0, got {self.num_matched}")

        # LightGlue early-exit / token-pruning: -1 disables, otherwise must lie in (0, 1).
        for name, val in (
            ("lightglue_depth_confidence", self.lightglue_depth_confidence),
            ("lightglue_width_confidence", self.lightglue_width_confidence),
        ):
            if val != -1 and not 0 < val < 1:
                raise ValueError(
                    f"{name} must be -1 (disable) or in (0, 1), got {val}"
                )

        valid_mappers = {"incremental", "global"}
        if self.mapper not in valid_mappers:
            raise ValueError(
                f"Invalid mapper {self.mapper!r}. Valid: {sorted(valid_mappers)}"
            )

        valid_camera_modes = {n.lower() for n in pycolmap.CameraMode.__members__}
        if self.camera_mode not in valid_camera_modes:
            raise ValueError(
                f"Invalid camera_mode {self.camera_mode!r}. "
                f"Valid: {sorted(valid_camera_modes)}"
            )
        valid_camera_models = {
            n.lower() for n in pycolmap.CameraModelId.__members__
        } - {"invalid"}
        if self.camera_model not in valid_camera_models:
            raise ValueError(
                f"Invalid camera_model {self.camera_model!r}. "
                f"Valid: {sorted(valid_camera_models)}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ColmapConfig:
        raw = _load_yaml(path)
        colmap = raw.get("colmap", {})
        return cls(
            project_dir=raw["project_dir"],
            extractor=colmap["extractor"],
            matcher=colmap["matcher"],
            max_num_features=colmap["max_num_features"],
            resize_max=colmap["resize_max"],
            resize_force=colmap["resize_force"],
            nms_radius=colmap["nms_radius"],
            pair_strategy=colmap["pair_strategy"],
            vocab_tree_path=colmap["vocab_tree_path"],
            num_images=colmap["num_images"],
            num_nearest_neighbors=colmap["num_nearest_neighbors"],
            retrieval_model=colmap["retrieval_model"],
            num_matched=colmap["num_matched"],
            lightglue_depth_confidence=colmap["lightglue_depth_confidence"],
            lightglue_width_confidence=colmap["lightglue_width_confidence"],
            mapper=colmap["mapper"],
            glomap_refine=colmap["glomap_refine"],
            camera_mode=colmap["camera_mode"],
            camera_model=colmap["camera_model"],
        )



@dataclass
class GausMvsNetConfig:
    """Gaus-MVSNet depth inference configuration."""

    project_dir: Path
    checkpoint: str
    num_view: int
    num_depth: int
    max_h: int
    max_w: int
    interval_scale: float
    ndepths: str
    depth_inter_r: str
    cr_base_chs: str
    view_selection_theta0: float
    view_selection_sigma1: float
    view_selection_sigma2: float
    prob_threshold: float
    num_consistent: int
    reproj_threshold: float
    depth_threshold: float
    downsample_target: int
    downsample_max_iter: int
    downsample_tolerance: float
    downsample_batch_size: int

    # Derived paths
    output_dir: Path = field(init=False)
    colmap_dense: Path = field(init=False)
    colmap_images: Path = field(init=False)
    colmap_sparse: Path = field(init=False)
    mvs_dir: Path = field(init=False)
    mvs_cams_dir: Path = field(init=False)
    mvs_pair_file: Path = field(init=False)
    depth_dir: Path = field(init=False)
    confidence_dir: Path = field(init=False)
    filtered_depth_dir: Path = field(init=False)
    dmap_dir: Path = field(init=False)
    final_pointcloud: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.output_dir = self.project_dir / "output"
        self.colmap_dense = self.output_dir / "colmap" / "dense"
        self.colmap_images = self.colmap_dense / "images"
        self.colmap_sparse = self.colmap_dense / "sparse"
        self.mvs_dir = self.output_dir / "gaus_mvsnet"
        self.mvs_cams_dir = self.mvs_dir / "cams"
        self.mvs_pair_file = self.mvs_dir / "pair.txt"
        self.depth_dir = self.mvs_dir / "depth_est"
        self.confidence_dir = self.mvs_dir / "confidence"
        self.filtered_depth_dir = self.mvs_dir / "filtered_depth"
        self.dmap_dir = self.mvs_dir / "dmaps"
        self.final_pointcloud = self.mvs_dir / "final_pointcloud.ply"

        stages = (len(self.ndepths_list), len(self.depth_inter_r_list), len(self.cr_base_chs_list))
        if len(set(stages)) != 1:
            raise ValueError(
                f"Cascade stage counts disagree: ndepths={stages[0]}, "
                f"depth_inter_r={stages[1]}, cr_base_chs={stages[2]}"
            )
        if self.num_view < 2:
            raise ValueError(f"num_view must be >= 2, got {self.num_view}")
        if self.num_consistent < 1:
            raise ValueError(f"num_consistent must be >= 1, got {self.num_consistent}")
        if not 0.0 <= self.prob_threshold <= 1.0:
            raise ValueError(f"prob_threshold must be in [0, 1], got {self.prob_threshold}")
        # No 32-grid requirement here: the inference size is floored onto that
        # grid per image, so these are a cap rather than the size itself.
        for name, value in (("max_h", self.max_h), ("max_w", self.max_w)):
            if value < 32:
                raise ValueError(f"{name} must be at least 32, got {value}")

    @property
    def ndepths_list(self) -> list[int]:
        return [int(x) for x in self.ndepths.split(",") if x]

    @property
    def depth_inter_r_list(self) -> list[float]:
        return [float(x) for x in self.depth_inter_r.split(",") if x]

    @property
    def cr_base_chs_list(self) -> list[int]:
        return [int(x) for x in self.cr_base_chs.split(",") if x]

    @property
    def num_stage(self) -> int:
        return len(self.ndepths_list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GausMvsNetConfig:
        raw = _load_yaml(path)
        g = raw["gaus_mvsnet"]
        return cls(project_dir=raw["project_dir"], **{k: g[k] for k in _GAUS_MVSNET_KEYS})


_GAUS_MVSNET_KEYS = (
    "checkpoint",
    "num_view", "num_depth", "max_h", "max_w", "interval_scale",
    "ndepths", "depth_inter_r", "cr_base_chs",
    "view_selection_theta0", "view_selection_sigma1", "view_selection_sigma2",
    "prob_threshold", "num_consistent", "reproj_threshold", "depth_threshold",
    "downsample_target", "downsample_max_iter",
    "downsample_tolerance", "downsample_batch_size",
)


@dataclass
class OpenMVSConfig:
    """OpenMVS dense reconstruction configuration."""

    project_dir: Path
    mesh_only: bool
    bin_dir: str
    resolution_level: int
    min_resolution: int
    max_resolution: int
    number_views: int
    number_views_fuse: int
    estimate_colors: int
    estimate_normals: int
    mesh_decimate: float
    mesh_remove_spurious: int
    mesh_smooth: int
    cli_timeout: int
    omp_num_threads: int
    downsample_target: int
    downsample_max_iter: int
    downsample_tolerance: float
    downsample_batch_size: int

    # Derived paths
    output_dir: Path = field(init=False)
    colmap_dense: Path = field(init=False)
    openmvs_dir: Path = field(init=False)
    openmvs_logs_dir: Path = field(init=False)
    openmvs_scene: Path = field(init=False)
    openmvs_dense_mvs: Path = field(init=False)
    openmvs_dense_ply: Path = field(init=False)
    openmvs_mesh_mvs: Path = field(init=False)
    openmvs_mesh_ply: Path = field(init=False)
    final_pointcloud: Path = field(init=False)
    dmap_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.output_dir = self.project_dir / "output"
        self.colmap_dense = self.output_dir / "colmap" / "dense"
        self.dmap_dir = self.output_dir / "gaus_mvsnet" / "dmaps"
        self.openmvs_dir = self.output_dir / "openmvs"
        self.openmvs_logs_dir = self.openmvs_dir / "logs"
        self.openmvs_scene = self.openmvs_dir / "scene.mvs"
        self.openmvs_dense_mvs = self.openmvs_dir / "scene_dense.mvs"
        self.openmvs_dense_ply = self.openmvs_dir / "scene_dense.ply"
        self.openmvs_mesh_mvs = self.openmvs_dir / "scene_mesh.mvs"
        self.openmvs_mesh_ply = self.openmvs_dir / "scene_mesh.ply"
        self.final_pointcloud = self.openmvs_dir / "final_pointcloud.ply"

    @classmethod
    def from_yaml(cls, path: str | Path) -> OpenMVSConfig:
        raw = _load_yaml(path)
        o = raw["openmvs"]
        return cls(
            project_dir=raw["project_dir"],
            mesh_only=o["mesh_only"],
            bin_dir=o["bin_dir"],
            resolution_level=o["resolution_level"],
            min_resolution=o["min_resolution"],
            max_resolution=o["max_resolution"],
            number_views=o["number_views"],
            number_views_fuse=o["number_views_fuse"],
            estimate_colors=o["estimate_colors"],
            estimate_normals=o["estimate_normals"],
            mesh_decimate=o["mesh_decimate"],
            mesh_remove_spurious=o["mesh_remove_spurious"],
            mesh_smooth=o["mesh_smooth"],
            cli_timeout=o["cli_timeout"],
            omp_num_threads=o["omp_num_threads"],
            downsample_target=o["downsample_target"],
            downsample_max_iter=o["downsample_max_iter"],
            downsample_tolerance=o["downsample_tolerance"],
            downsample_batch_size=o["downsample_batch_size"],
        )


@dataclass
class VisualizationConfig:
    """Point-cloud/mask visualization parameters."""

    alpha: float
    beta: float
    color_seed: int


@dataclass
class QsSam3dConfig:
    """QS-SAM3D segmentation configuration."""

    project_dir: Path
    sam_model_path: str
    voxel_size: float
    min_group_size: int
    smooth_k: int
    smooth_iters: int
    image_size: tuple[int, int]
    quality_window_size: int
    frame_rejection_percent: int
    depth_shift: float
    match_distance_factor: int
    overlap_ratio: float
    distance_threshold: float
    fusion_consistency_threshold: float
    points_per_side: int
    pred_iou_thresh: float
    stability_score_thresh: float
    crop_n_layers: int
    crop_n_points_downscale_factor: int
    min_mask_region_area: int
    viz: VisualizationConfig

    # Derived paths
    output_dir: Path = field(init=False)
    colmap_dense: Path = field(init=False)
    colmap_images: Path = field(init=False)
    openmvs_dir: Path = field(init=False)
    openmvs_mesh_ply: Path = field(init=False)
    final_pointcloud: Path = field(init=False)
    sam3d_dir: Path = field(init=False)
    sam3d_color_dir: Path = field(init=False)
    sam3d_depth_dir: Path = field(init=False)
    sam3d_pose_dir: Path = field(init=False)
    sam3d_intrinsics_dir: Path = field(init=False)
    sam3d_intrinsic_global: Path = field(init=False)
    sam3d_pointcloud: Path = field(init=False)
    sam3d_seg_dir: Path = field(init=False)
    sam3d_segmentation: Path = field(init=False)
    sam3d_segmentation_ply: Path = field(init=False)
    sam3d_mesh_segmentation_ply: Path = field(init=False)
    sam3d_masks_dir: Path = field(init=False)
    sam3d_merged_cloud: Path = field(init=False)
    sam3d_viz_dir: Path = field(init=False)
    sam3d_segmented_mesh_ply: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.output_dir = self.project_dir / "output"
        self.colmap_dense = self.output_dir / "colmap" / "dense"
        self.colmap_images = self.colmap_dense / "images"
        self.openmvs_dir = self.output_dir / "openmvs"
        self.openmvs_mesh_ply = self.openmvs_dir / "scene_mesh.ply"
        self.final_pointcloud = self.output_dir / "gaus_mvsnet" / "final_pointcloud.ply"
        self.sam3d_dir = self.output_dir / "qs_sam3d"
        self.sam3d_color_dir = self.sam3d_dir / "color"
        self.sam3d_depth_dir = self.sam3d_dir / "depth"
        self.sam3d_pose_dir = self.sam3d_dir / "pose"
        self.sam3d_intrinsics_dir = self.sam3d_dir / "intrinsics"
        self.sam3d_intrinsic_global = self.sam3d_intrinsics_dir / "intrinsic_depth.txt"
        self.sam3d_pointcloud = self.sam3d_dir / "pointcloud.pth"
        self.sam3d_seg_dir = self.sam3d_dir / "segmentation"
        self.sam3d_segmentation = self.sam3d_seg_dir / "labels.pth"
        self.sam3d_segmentation_ply = self.sam3d_seg_dir / "labels.ply"
        self.sam3d_mesh_segmentation_ply = self.sam3d_seg_dir / "mesh_labels.ply"
        self.sam3d_masks_dir = self.sam3d_seg_dir / "masks"
        self.sam3d_merged_cloud = self.sam3d_seg_dir / "merged_cloud.ply"
        self.sam3d_viz_dir = self.sam3d_seg_dir / "visualization"
        self.sam3d_segmented_mesh_ply = self.sam3d_viz_dir / "segmented_mesh.ply"

    @classmethod
    def from_yaml(cls, path: str | Path) -> QsSam3dConfig:
        raw = _load_yaml(path)
        s = raw["qs_sam3d"]
        return cls(
            project_dir=raw["project_dir"],
            sam_model_path=s["sam_model_path"],
            voxel_size=s["voxel_size"],
            min_group_size=s["min_group_size"],
            smooth_k=s["smooth_k"],
            smooth_iters=s["smooth_iters"],
            image_size=tuple(s["image_size"]),
            quality_window_size=s["quality_window_size"],
            frame_rejection_percent=s["frame_rejection_percent"],
            depth_shift=s["depth_shift"],
            match_distance_factor=s["match_distance_factor"],
            overlap_ratio=s["overlap_ratio"],
            distance_threshold=s["distance_threshold"],
            fusion_consistency_threshold=s["fusion_consistency_threshold"],
            points_per_side=s["points_per_side"],
            pred_iou_thresh=s["pred_iou_thresh"],
            stability_score_thresh=s["stability_score_thresh"],
            crop_n_layers=s["crop_n_layers"],
            crop_n_points_downscale_factor=s["crop_n_points_downscale_factor"],
            min_mask_region_area=s["min_mask_region_area"],
            viz=_viz_from_yaml(raw),
        )


# ---------------------------------------------------------------------------
# Pipeline config (aggregates all module configs)
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    project_dir: Path
    colmap: ColmapConfig
    gaus_mvsnet: GausMvsNetConfig
    openmvs: OpenMVSConfig
    qs_sam3d: QsSam3dConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        raw = _load_yaml(path)
        project_dir = Path(raw["project_dir"])
        return cls(
            project_dir=project_dir,
            colmap=ColmapConfig(
                project_dir=project_dir,
                extractor=raw["colmap"]["extractor"],
                matcher=raw["colmap"]["matcher"],
                max_num_features=raw["colmap"]["max_num_features"],
                resize_max=raw["colmap"]["resize_max"],
                resize_force=raw["colmap"]["resize_force"],
                nms_radius=raw["colmap"]["nms_radius"],
                pair_strategy=raw["colmap"]["pair_strategy"],
                vocab_tree_path=raw["colmap"]["vocab_tree_path"],
                num_images=raw["colmap"]["num_images"],
                num_nearest_neighbors=raw["colmap"]["num_nearest_neighbors"],
                retrieval_model=raw["colmap"]["retrieval_model"],
                num_matched=raw["colmap"]["num_matched"],
                lightglue_depth_confidence=raw["colmap"]["lightglue_depth_confidence"],
                lightglue_width_confidence=raw["colmap"]["lightglue_width_confidence"],
                mapper=raw["colmap"]["mapper"],
                glomap_refine=raw["colmap"]["glomap_refine"],
                camera_mode=raw["colmap"]["camera_mode"],
                camera_model=raw["colmap"]["camera_model"],
            ),
            gaus_mvsnet=GausMvsNetConfig(
                project_dir=project_dir,
                **{k: raw["gaus_mvsnet"][k] for k in _GAUS_MVSNET_KEYS},
            ),
            openmvs=OpenMVSConfig(
                project_dir=project_dir,
                **{k: raw["openmvs"][k] for k in (
                    "mesh_only", "bin_dir",
                    "resolution_level", "min_resolution", "max_resolution",
                    "number_views", "number_views_fuse",
                    "estimate_colors", "estimate_normals",
                    "mesh_decimate", "mesh_remove_spurious", "mesh_smooth",
                    "cli_timeout", "omp_num_threads",
                    "downsample_target", "downsample_max_iter",
                    "downsample_tolerance", "downsample_batch_size",
                )},
            ),
            qs_sam3d=QsSam3dConfig(
                project_dir=project_dir,
                sam_model_path=raw["qs_sam3d"]["sam_model_path"],
                voxel_size=raw["qs_sam3d"]["voxel_size"],
                min_group_size=raw["qs_sam3d"]["min_group_size"],
                smooth_k=raw["qs_sam3d"]["smooth_k"],
                smooth_iters=raw["qs_sam3d"]["smooth_iters"],
                image_size=tuple(raw["qs_sam3d"]["image_size"]),
                quality_window_size=raw["qs_sam3d"]["quality_window_size"],
                frame_rejection_percent=raw["qs_sam3d"]["frame_rejection_percent"],
                depth_shift=raw["qs_sam3d"]["depth_shift"],
                match_distance_factor=raw["qs_sam3d"]["match_distance_factor"],
                overlap_ratio=raw["qs_sam3d"]["overlap_ratio"],
                distance_threshold=raw["qs_sam3d"]["distance_threshold"],
                fusion_consistency_threshold=raw["qs_sam3d"]["fusion_consistency_threshold"],
                points_per_side=raw["qs_sam3d"]["points_per_side"],
                pred_iou_thresh=raw["qs_sam3d"]["pred_iou_thresh"],
                stability_score_thresh=raw["qs_sam3d"]["stability_score_thresh"],
                crop_n_layers=raw["qs_sam3d"]["crop_n_layers"],
                crop_n_points_downscale_factor=raw["qs_sam3d"]["crop_n_points_downscale_factor"],
                min_mask_region_area=raw["qs_sam3d"]["min_mask_region_area"],
                viz=_viz_from_yaml(raw),
            ),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _viz_from_yaml(raw: dict) -> VisualizationConfig:
    v = raw["visualization"]
    return VisualizationConfig(
        alpha=v["alpha"], beta=v["beta"], color_seed=v["color_seed"],
    )


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    project_dir = Path(raw["project_dir"])
    if not project_dir.is_absolute():
        project_dir = (path.parent / project_dir).resolve()
    raw["project_dir"] = project_dir

    sam_path = raw.get("qs_sam3d", {}).get("sam_model_path")
    if sam_path is not None and not Path(sam_path).is_absolute():
        raw["qs_sam3d"]["sam_model_path"] = str((path.parent / sam_path).resolve())

    ckpt = raw.get("gaus_mvsnet", {}).get("checkpoint")
    if ckpt and not Path(ckpt).is_absolute():
        raw["gaus_mvsnet"]["checkpoint"] = str((path.parent / ckpt).resolve())

    bin_dir = raw.get("openmvs", {}).get("bin_dir")
    if bin_dir and not Path(bin_dir).is_absolute():
        raw["openmvs"]["bin_dir"] = str((path.parent / bin_dir).resolve())

    vocab_tree_path = raw.get("colmap", {}).get("vocab_tree_path")
    if vocab_tree_path and not Path(vocab_tree_path).is_absolute():
        raw["colmap"]["vocab_tree_path"] = str((path.parent / vocab_tree_path).resolve())
    return raw
