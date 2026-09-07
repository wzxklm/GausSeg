# GausSeg

Class-agnostic 3D instance segmentation from a folder of RGB images. The
pipeline runs Structure-from-Motion, dense reconstruction, and SAM-based mask
lifting, and writes a labelled point cloud and mesh.

> **Looking for Gaus-MVSNet?** This branch recovers depth with OpenMVS
> PatchMatch. The pipeline of the paper — depth from the Gaus-MVSNet network,
> with OpenMVS reduced to meshing — lives on the
> [`gaus-mvsnet`](../../tree/gaus-mvsnet) branch, along with its DTU
> checkpoint. The two branches are kept separate; neither is merged into the
> other.

## Pipeline

| # | Stage | Module | Writes |
| --- | --- | --- | --- |
| 1 | COLMAP SfM | `colmap/sfm.py` | Camera poses, sparse cloud |
| 2 | OpenMVS Dense | `openmvs/dense.py` | Dense cloud, mesh, downsampled cloud |
| 3 | QS-SAM3D Convert | `qs_sam3d/converter.py` | Per-view RGB-D + intrinsics for SAM |
| 4 | QS-SAM3D Segment | `qs_sam3d/segmentation.py` | Labelled point cloud, labelled mesh |

Stage 4 reads stage 2's `scene_mesh.ply`, so `--mode segment` cannot be run
against a scene that has not been through stage 2.

---

## Installation

COLMAP 4.0.3 and OpenMVS 2.4.0 have to be compiled against CUDA, so the
repository ships a devcontainer that does it.

**Prerequisites**

- NVIDIA GPU, `docker`, `docker compose`, `nvidia-container-toolkit`
- The image hardcodes CUDA arch `sm_89` (Ada). On other GPUs, edit
  `TORCH_CUDA_ARCH_LIST` and `CMAKE_CUDA_ARCHITECTURES` in
  [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile).

**VS Code** — install the *Dev Containers* extension, open the repository, then
*Dev Containers: Reopen in Container*.

**Command line**

```bash
cd .devcontainer
docker compose build          # 30-60 min: glog, Ceres, COLMAP, OpenMVS compile from source
docker compose up -d
docker compose exec gausseg bash
```

Verify the environment:

```bash
python -c "import pycolmap, torch, pointops; from hloc.extractors.superpoint import SuperPoint; print('ok')"
InterfaceCOLMAP -h >/dev/null && echo "OpenMVS ok"
```

<details>
<summary><b>Installing without Docker</b></summary>

There is no `requirements.txt` — pip cannot cover this stack, and the Dockerfile
is the authoritative build. Installing by hand needs:

- **System**: CUDA 12.4 toolkit, glog ≥ 0.7.1 *(apt's 0.4.0 `CHECK_OP_LOG` macro
  ignores custom handlers, which makes pycolmap abort on init)*, Ceres with
  cuDSS, COLMAP 4.0.3 built with `ONNX_ENABLED`, OpenMVS 2.4.0,
  nanoflann ≥ 1.6.3, CGAL 6.0.1
- **Python**: `torch==2.4.0` (cu124), `torchvision==0.19.0`, `numpy`,
  `opencv-python`, `open3d`, `scipy`, `Pillow`, `matplotlib`, `pyyaml`, `tqdm`,
  `safetensors`
- **From source**: `pycolmap` 4.0.3 (COLMAP's own bindings — the PyPI wheel will
  not do), [hloc](https://github.com/cvg/Hierarchical-Localization),
  [segment-anything](https://github.com/facebookresearch/segment-anything),
  and `pointops` (see [`pointops/README.md`](pointops/README.md))

Read the [Dockerfile](.devcontainer/Dockerfile) first; its comments record the
version pins and patches each of these needs.

</details>

---

## Input

**Scene layout.** Point `project_dir` in [`config.yaml`](config.yaml) at a
directory shaped like this. Only `inputs/images/` is yours to create; the rest
appears as stages run.

```
data/<scene>/
├── inputs/
│   └── images/          # your *.jpg / *.png
└── output/
    ├── colmap/          # stage 1
    ├── openmvs/         # stage 2
    └── qs_sam3d/        # stages 3-4
```

**SAM weights.** Not redistributed here. Fetch ViT-H (~2.4 GB) to wherever
`qs_sam3d.sam_model_path` points:

```bash
wget -O qs_sam3d/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Running

```bash
python pipeline.py --config config.yaml --mode all
```

| `--mode` | Stages |
| --- | --- |
| `reconstruct` | 1 → 2 |
| `segment` | 3 → 4 |
| `all` *(default)* | 1 → 2 → 3 → 4 |

CUDA memory is released between stages, so a full run does not accumulate.

Any stage also runs standalone, reading `config.yaml` from the repository root
unless given `--config`:

```bash
python -m colmap.sfm
python -m openmvs.dense
python -m qs_sam3d.converter
python -m qs_sam3d.segmentation
```

---

## Output

Under `<scene>/output/qs_sam3d/`:

| File | Contents |
| --- | --- |
| `segmentation/labels.ply` | Points coloured by instance |
| `segmentation/mesh_labels.ply` | Same labels on the OpenMVS mesh |
| `segmentation/merged_cloud.ply` | Pre-smoothing merge |
| `visualization/segmented_mesh.ply` | Labels alpha-blended over original colour |

Points further than `qs_sam3d.distance_threshold` from any segment are written
as background (label `-1`).

---

## Configuration

Everything lives in [`config.yaml`](config.yaml), commented inline. The settings
that need changing per scene:

**Stage 1 — scale to your image count**

| Setting | Guidance |
| --- | --- |
| `colmap.pair_strategy` | `exhaustive` under a few hundred images; `retrieval` (SuperPoint) or `vocab_tree` (SIFT) above that |
| `colmap.num_matched` | Retrieval neighbours per image: 10–20 up to ~500 images, 20–30 for 500–1k, 30–50 beyond |
| `colmap.mapper` | `incremental` by default; `global` (GLOMAP) from a few thousand images up |
| `colmap.max_num_features` | LightGlue attention is O(N²) — lower this first when VRAM-bound |

**Stage 2 — cost vs. quality**

| Setting | Guidance |
| --- | --- |
| `openmvs.resolution_level` | `0` = full res, `1` = half, `2` = quarter |
| `openmvs.number_views` | Source views per pixel |
| `openmvs.omp_num_threads` | Keep at `1` on 16 GB cards — the 16-way default has been observed to OOM on a 49-image scene |

**Stage 4**

| Setting | Guidance |
| --- | --- |
| `qs_sam3d.frame_rejection_percent` | Percentile of worst-scoring frames dropped before merging |
| `qs_sam3d.quality_window_size` | Neighbour window the frame score is computed over |
| `qs_sam3d.overlap_ratio` | IoU threshold for merging masks in 3D |
| `qs_sam3d.points_per_side`, `pred_iou_thresh` | SAM mask granularity |
| `qs_sam3d.smooth_k`, `smooth_iters` | KNN majority vote that cleans up small label islands |

---

## Tests

`tests/` holds one script per stage, run in order, each consuming the previous
stage's output. They do not mock: each constructs the real stage from
`config.yaml` and asserts its artifacts landed. A single-stage test doubles as a
minimum reproduction script when debugging.

```bash
python tests/test_all.py 2>&1 | tee tests/test_all.log
```

> Unlike `pipeline.py`, the test scripts do **not** accept `--config` — they read
> `config.yaml` from the repository root. Edit that file to point them at a
> different scene.

---

## Third-party code

[`pointops/`](pointops/) is a vendored copy of `libs/pointops` from
[Pointcept](https://github.com/Pointcept/Pointcept) (MIT, © 2023 Pointcept),
originally from Point Transformer V2. It is **not** covered by this
repository's license — see [`pointops/LICENSE`](pointops/LICENSE) and
[`pointops/README.md`](pointops/README.md) for provenance and the one local
modification.

The pipeline also builds on [COLMAP](https://github.com/colmap/colmap),
[GLOMAP](https://github.com/colmap/glomap),
[OpenMVS](https://github.com/cdcseacave/openMVS),
[hloc](https://github.com/cvg/Hierarchical-Localization),
[LightGlue](https://github.com/cvg/LightGlue) and
[Segment Anything](https://github.com/facebookresearch/segment-anything), each
under its own license.

## License

[MIT](LICENSE), except `pointops/` as noted above.
