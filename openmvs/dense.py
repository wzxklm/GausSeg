"""OpenMVS dense reconstruction: subprocess CLI wrapper.

Pipeline: InterfaceCOLMAP -> DensifyPointCloud -> ReconstructMesh ->
per-vertex mesh colorization (KNN from dense cloud) -> voxel downsample.

Under `mesh_only` the PatchMatch estimation is skipped and DensifyPointCloud
only fuses the depth maps Gaus-MVSNet already wrote, which is what the paper's
pipeline does: depth comes from the network, and OpenMVS contributes nothing
but the surface the labels are later drawn on.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm.auto import tqdm

from config import OpenMVSConfig
from utils.depth import read_dmap
from utils.pointcloud import voxel_downsample

logger = logging.getLogger(__name__)

# Each OpenMVS phase reports progress one line at a time as
# `<phrase> N (XX.XX%, <elapsed>, ETA <eta>)...`, closing with a
# `<phrase> N (100%, <total>)` line. Five phases do this (Estimated /
# Geometric-consistent / Dense fused depth-maps, Points inserted / weighted).
# Collapse those per-line INFOs into one tqdm bar and keep only the 100% summary.
_PROGRESS_RE = re.compile(
    r"^(?P<phase>.+?) (?P<n>\d+) \((?P<pct>\d+(?:\.\d+)?)%, [^,]+, ETA [^)]+\)\.\.\.$"
)
_PROGRESS_DONE_RE = re.compile(
    r"^(?P<phase>.+?) (?P<n>\d+) \(100(?:\.0+)?%, [^)]+\)$"
)


class OpenMVSDense:
    """OpenMVS dense reconstruction pipeline via CLI subprocess."""

    def __init__(self, config: OpenMVSConfig) -> None:
        self.config = config

    def run(self) -> None:
        if not self.config.colmap_dense.is_dir():
            raise FileNotFoundError(f"COLMAP dense not found: {self.config.colmap_dense}")

        if self.config.openmvs_dir.exists():
            shutil.rmtree(self.config.openmvs_dir)
        self.config.openmvs_dir.mkdir(parents=True)

        self._interface_colmap()
        if self.config.mesh_only:
            self._link_dmaps()
            self._fuse_dmaps()
        else:
            self._densify()
        self._reconstruct_mesh()
        self._colorize_mesh()
        self._downsample()

        logger.info("OpenMVS dense complete: %s", self.config.openmvs_dir)

    def _interface_colmap(self) -> None:
        """COLMAP dense/ -> scene.mvs"""
        self._run("InterfaceCOLMAP", [
            "-w", str(self.config.openmvs_dir),
            "-i", str(self.config.colmap_dense),
            "-o", self.config.openmvs_scene.name,
        ])
        if not self.config.openmvs_scene.is_file():
            raise RuntimeError(f"InterfaceCOLMAP produced no {self.config.openmvs_scene}")

    def _densify(self) -> None:
        """scene.mvs -> scene_dense.mvs + scene_dense.ply"""
        self._run("DensifyPointCloud", [
            "-w", str(self.config.openmvs_dir),
            "-i", self.config.openmvs_scene.name,
            "-o", self.config.openmvs_dense_mvs.name,
            "--resolution-level", str(self.config.resolution_level),
            "--min-resolution", str(self.config.min_resolution),
            "--max-resolution", str(self.config.max_resolution),
            "--number-views", str(self.config.number_views),
            "--number-views-fuse", str(self.config.number_views_fuse),
            "--estimate-colors", str(self.config.estimate_colors),
            "--estimate-normals", str(self.config.estimate_normals),
        ])
        if not self.config.openmvs_dense_ply.is_file():
            raise RuntimeError(f"DensifyPointCloud produced no {self.config.openmvs_dense_ply}")

    def _link_dmaps(self) -> None:
        """Expose Gaus-MVSNet's depth maps in the OpenMVS working folder.

        Symlinked rather than copied: at source resolution these run to
        hundreds of MB per scene, and OpenMVS only ever reads them.
        """
        dmaps = sorted(self.config.dmap_dir.glob("depth*.dmap"))
        if not dmaps:
            raise FileNotFoundError(
                f"No .dmap files in {self.config.dmap_dir}. "
                "Run the gaus_mvsnet stage before openmvs in mesh_only mode."
            )
        for dmap in dmaps:
            (self.config.openmvs_dir / dmap.name).symlink_to(dmap.resolve())
        logger.info("Linked %d depth maps from %s", len(dmaps), self.config.dmap_dir)

    def _fuse_dmaps(self) -> None:
        """scene.mvs + linked .dmap -> scene_dense.mvs + scene_dense.ply

        OpenMVS indexes a depth map by pixel into the image it loaded itself
        and segfaults when the two disagree, so its resolution is pinned to the
        one the maps were written at instead of being left to the
        resolution-level and min/max clamps.
        """
        resolution = self._dmap_resolution()
        self._run("DensifyPointCloud", [
            "-w", str(self.config.openmvs_dir),
            "-i", self.config.openmvs_scene.name,
            "-o", self.config.openmvs_dense_mvs.name,
            "--resolution-level", "0",
            "--min-resolution", str(resolution),
            "--max-resolution", str(resolution),
            "--number-views", str(self.config.number_views),
            "--number-views-fuse", str(self.config.number_views_fuse),
            "--estimate-colors", str(self.config.estimate_colors),
            "--estimate-normals", str(self.config.estimate_normals),
            # Refining the maps against their neighbours would overwrite the
            # network's depth with PatchMatch's.
            "--geometric-iters", "0",
        ])
        if not self.config.openmvs_dense_ply.is_file():
            raise RuntimeError(f"DensifyPointCloud produced no {self.config.openmvs_dense_ply}")

    def _dmap_resolution(self) -> int:
        sizes = {
            (d["image_width"], d["image_height"])
            for d in (read_dmap(p) for p in sorted(self.config.dmap_dir.glob("depth*.dmap")))
        }
        if len(sizes) > 1:
            raise RuntimeError(
                f"Depth maps disagree on resolution ({sorted(sizes)}); OpenMVS "
                "fuses a single pinned resolution per scene"
            )
        width, height = sizes.pop()
        return max(width, height)

    def _reconstruct_mesh(self) -> None:
        """scene_dense.mvs -> scene_mesh.mvs + scene_mesh.ply"""
        self._run("ReconstructMesh", [
            "-w", str(self.config.openmvs_dir),
            "-i", self.config.openmvs_dense_mvs.name,
            "-o", self.config.openmvs_mesh_ply.name,
            "--decimate", str(self.config.mesh_decimate),
            "--remove-spurious", str(self.config.mesh_remove_spurious),
            "--smooth", str(self.config.mesh_smooth),
        ])
        if not self.config.openmvs_mesh_ply.is_file():
            raise RuntimeError(f"ReconstructMesh produced no {self.config.openmvs_mesh_ply}")

    def _colorize_mesh(self) -> None:
        """Transfer per-vertex colors to scene_mesh.ply from the dense point cloud.

        ReconstructMesh builds a Delaunay mesh whose vertices are not the input
        dense points, so colors are lost. Assign each vertex the color of its
        nearest dense-point neighbor.
        """
        mesh = o3d.io.read_triangle_mesh(str(self.config.openmvs_mesh_ply))
        dense = o3d.io.read_point_cloud(str(self.config.openmvs_dense_ply))
        if not dense.has_colors():
            logger.warning("Dense point cloud has no colors; skipping mesh colorization")
            return

        verts = np.asarray(mesh.vertices)
        points = np.asarray(dense.points)
        colors = np.asarray(dense.colors)

        _, idx = KDTree(points).query(verts, k=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx].astype(np.float64))

        o3d.io.write_triangle_mesh(
            str(self.config.openmvs_mesh_ply), mesh, write_ascii=False
        )
        logger.info("Colorized mesh: %d vertices", len(verts))

    def _downsample(self) -> None:
        """scene_dense.ply -> final_pointcloud.ply (voxel-grid downsample)."""
        voxel_downsample(
            input_path=self.config.openmvs_dense_ply,
            output_path=self.config.final_pointcloud,
            target_count=self.config.downsample_target,
            max_iter=self.config.downsample_max_iter,
            tolerance=self.config.downsample_tolerance,
            batch_size=self.config.downsample_batch_size,
        )

    def _run(self, tool: str, args: list[str]) -> None:
        """Invoke an OpenMVS CLI tool with streaming logs and timeout.

        OpenMVS writes `<tool>-<timestamp>.log` to the `-w` folder. After the
        tool exits, we relocate any fresh log into `openmvs_dir/logs/` to keep
        the scene directory clean.
        """
        exe = self._resolve_tool(tool)
        cmd = [exe, *args]
        logger.info("[%s] %s", tool, " ".join(cmd))

        # Cap OMP concurrency so DensifyPointCloud's parallel PatchMatch
        # workers don't exhaust VRAM/CUDA handles on dense scenes.
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.config.omp_num_threads)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc.stdout is None:
            raise RuntimeError(f"{tool}: failed to open subprocess stdout pipe")

        bar: tqdm | None = None
        cur_phase: str | None = None
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                m = _PROGRESS_RE.match(line)
                if m:
                    phase = m.group("phase")
                    if phase != cur_phase:
                        if bar is not None:
                            bar.close()
                        bar = tqdm(total=100.0, desc=f"[{tool}] {phase}", unit="%")
                        cur_phase = phase
                    bar.n = min(float(m.group("pct")), 100.0)
                    bar.refresh()
                    continue
                m = _PROGRESS_DONE_RE.match(line)
                if m and bar is not None and m.group("phase") == cur_phase:
                    bar.n = 100.0
                    bar.refresh()
                    bar.close()
                    bar = None
                    cur_phase = None
                    logger.info("[%s] %s", tool, line)
                    continue
                if bar is not None:
                    bar.close()
                    bar = None
                    cur_phase = None
                logger.info("[%s] %s", tool, line)
            proc.wait(timeout=self.config.cli_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"{tool} timed out after {self.config.cli_timeout}s")
        finally:
            if bar is not None:
                bar.close()

        self._relocate_logs(tool)

        if proc.returncode != 0:
            raise RuntimeError(f"{tool} failed (exit {proc.returncode})")

    def _resolve_tool(self, tool: str) -> str:
        """Resolve an OpenMVS CLI tool to an absolute path.

        Uses `openmvs.bin_dir` when set so builds can pin a specific install;
        falls back to PATH lookup otherwise.
        """
        if self.config.bin_dir:
            exe = Path(self.config.bin_dir) / tool
            if not exe.is_file():
                raise FileNotFoundError(
                    f"OpenMVS tool not found in bin_dir={self.config.bin_dir}: {tool}"
                )
            return str(exe)
        exe_path = shutil.which(tool)
        if exe_path is None:
            raise FileNotFoundError(f"OpenMVS tool not found on PATH: {tool}")
        return exe_path

    def _relocate_logs(self, tool: str) -> None:
        """Move OpenMVS-generated <tool>-*.log files from openmvs_dir to logs/."""
        self.config.openmvs_logs_dir.mkdir(parents=True, exist_ok=True)
        for log in self.config.openmvs_dir.glob(f"{tool}-*.log"):
            log.replace(self.config.openmvs_logs_dir / log.name)


if __name__ == "__main__":
    from utils.cli import run_stage

    run_stage(__file__, "OpenMVS dense reconstruction", OpenMVSConfig.from_yaml, OpenMVSDense)
