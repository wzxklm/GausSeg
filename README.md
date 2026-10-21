# GausSeg — `gaus-mvsnet` branch

Class-agnostic 3D instance segmentation from a folder of RGB images. The
pipeline runs Structure-from-Motion, learned multi-view stereo, and SAM-based
mask lifting, and writes a labelled point cloud and mesh.

> **This branch replaces the depth source.** On `main`, depth comes from
> OpenMVS PatchMatch. Here it comes from Gaus-MVSNet, the unsupervised MVS
> network of the paper, and OpenMVS is reduced to building the surface the
> labels are drawn on. The two branches are kept separate rather than merged.
>
> **Inference only.** The 3D Gaussian Splatting auxiliary branch exists to
> supervise training and plays no part in inference, so it is not on this
> branch. The training code is on the `gaus-mvsnet-train` branch, and the
> checkpoints it writes load here unchanged.

## Pipeline

| # | Stage | Module | Writes |
| --- | --- | --- | --- |
| 1 | COLMAP SfM | `colmap/sfm.py` | Camera poses, sparse cloud |
| 2 | Gaus-MVSNet Depth | `gaus_mvsnet/depth.py` | Per-view depth (`.dmap`), fused point cloud |
| 3 | OpenMVS Mesh | `openmvs/dense.py` | Surface mesh, fused dense cloud |
| 4 | QS-SAM3D Convert | `qs_sam3d/converter.py` | Per-view RGB-D + intrinsics for SAM |
| 5 | QS-SAM3D Segment | `qs_sam3d/segmentation.py` | Labelled point cloud, labelled mesh |

Stage 2 predicts one depth map per view, filters them against each other, and
fuses the survivors into `final_pointcloud.ply` — the geometry stage 5
segments. It also writes each depth map in OpenMVS's `.dmap` format, which is
what stages 3 and 4 read.

Stage 3 does not run PatchMatch. With `openmvs.mesh_only` set (the default on
this branch) `DensifyPointCloud` only *fuses* the depth maps stage 2 wrote,
which takes seconds, and `ReconstructMesh` turns the result into the surface
stage 5 draws labels on. The labels themselves are computed on stage 2's point
cloud and transferred to mesh vertices by nearest neighbour, so the mesh
changes how the result looks, never what it says.

Stage 5 reads stage 3's `scene_mesh.ply`, so `--mode segment` cannot be run
against a scene that has not been through stages 2 and 3.

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

**Gaus-MVSNet weights.** No pretrained checkpoint is distributed. Train one on
DTU with the `gaus-mvsnet-train` branch (see its README), then point
`gaus_mvsnet.checkpoint` at one of the `model_XXXXXX.ckpt` files it writes:

```bash
mkdir -p gaus_mvsnet/checkpoint
cp <log_dir>/model_000014.ckpt gaus_mvsnet/checkpoint/gaus_mvsnet_dtu.ckpt
```

It is loaded with `strict=True`, so a checkpoint whose parameter names or
shapes differ from `gaus_mvsnet/model/casmvsnet.py` fails loudly at startup
rather than silently predicting from partly random weights.

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
| `depth` | 1 → 2 |
| `reconstruct` | 1 → 2 → 3 |
| `segment` | 4 → 5 |
| `all` *(default)* | 1 → 2 → 3 → 4 → 5 |

CUDA memory is released between stages, so a full run does not accumulate.

Any stage also runs standalone, reading `config.yaml` from the repository root
unless given `--config`:

```bash
python -m colmap.sfm
python -m gaus_mvsnet.depth
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

Everything lives in [`config.yaml`](config.yaml), commented inline. These
defaults are tuned for general capture rather than for benchmarks, so a few of
them differ from the values the paper reports; set these back to reproduce the
paper's numbers:

| Setting | Default here | Paper |
| --- | --- | --- |
| `qs_sam3d.frame_rejection_percent` | 10 | *ρ* = 15% |
| `qs_sam3d.min_group_size` | 100 | *τ_g* = 50 |
| `qs_sam3d.distance_threshold` | 0.5 | *τ_d* = 1.5·*v_s* = 0.0075 |
| `qs_sam3d.pred_iou_thresh` | 0.86 | 0.88 (SAM default) |
| `qs_sam3d.stability_score_thresh` | 0.92 | 0.95 (SAM default) |
| `qs_sam3d.crop_n_layers` | 1 | 0 (SAM default) |
| `qs_sam3d.crop_n_points_downscale_factor` | 2 | 1 (SAM default) |
| `qs_sam3d.min_mask_region_area` | 100 | 0 (SAM default) |
| `gaus_mvsnet.reproj_threshold` | 0.25 | *τ_p* = 1 px |
| `gaus_mvsnet.max_h` | 1200 | 1152, evaluating DTU at 1600×1152 |

Everything else already matches the paper. Note that `voxel_size` and
`distance_threshold` are scene-scale dependent: COLMAP reconstructions carry
no metric scale, so rescale both to the scene instead of copying them between
datasets.

The settings that need changing per scene:

**Stage 1 — scale to your image count**

| Setting | Guidance |
| --- | --- |
| `colmap.pair_strategy` | `exhaustive` under a few hundred images; `retrieval` (SuperPoint) or `vocab_tree` (SIFT) above that |
| `colmap.num_matched` | Retrieval neighbours per image: 10–20 up to ~500 images, 20–30 for 500–1k, 30–50 beyond |
| `colmap.mapper` | `incremental` by default; `global` (GLOMAP) from a few thousand images up |
| `colmap.max_num_features` | LightGlue attention is O(N²) — lower this first when VRAM-bound |

**Stage 2 — depth inference**

| Setting | Guidance |
| --- | --- |
| `gaus_mvsnet.checkpoint` | Weights trained on the `gaus-mvsnet-train` branch; see *Input* above |
| `gaus_mvsnet.max_h`, `max_w` | Inference resolution, both multiples of 32. The cost volume dominates VRAM — halve these first when it does not fit |
| `gaus_mvsnet.num_view` | Reference + source views per sample (paper: 5) |
| `gaus_mvsnet.num_depth` | Depth hypotheses at the coarsest stage |
| `gaus_mvsnet.prob_threshold`, `num_consistent`, `reproj_threshold`, `depth_threshold` | Fusion filter. Tighten them for a sparser, cleaner cloud |

**Stage 3 — mesh**

| Setting | Guidance |
| --- | --- |
| `openmvs.mesh_only` | `true` fuses stage 2's depth maps (seconds); `false` runs full PatchMatch densification and ignores them |
| `openmvs.resolution_level` | Only read when `mesh_only` is `false` — the fusion path is pinned to the resolution the depth maps were written at |

**Stage 3 — PatchMatch densification (only when `mesh_only: false`)**

| Setting | Guidance |
| --- | --- |
| `openmvs.resolution_level` | `0` = full res, `1` = half, `2` = quarter |
| `openmvs.number_views` | Source views per pixel |
| `openmvs.omp_num_threads` | Keep at `1` on 16 GB cards — the 16-way default has been observed to OOM on a 49-image scene |

**Stage 5**

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

[`gaus_mvsnet/model/`](gaus_mvsnet/model/) holds the cascade MVS backbone,
vendored from [CasMVSNet](https://github.com/alibaba/cascade-stereo) (MIT,
© 2019 Alibaba) in the form used by
[RC-MVSNet](https://github.com/Boese0601/RC-MVSNet) (MIT, © 2022 Di Chang),
and kept close to that form so checkpoints from the training branch load by
parameter name. It is covered by [`gaus_mvsnet/model/LICENSE`](gaus_mvsnet/model/LICENSE)
rather than this repository's; see
[`gaus_mvsnet/model/README.md`](gaus_mvsnet/model/README.md) for what was
changed. The rest of `gaus_mvsnet/` is this repository's own code.

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

[MIT](LICENSE), except `pointops/` and `gaus_mvsnet/model/` as noted above.
