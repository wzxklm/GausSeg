# GausSeg — `gaus-mvsnet-train` branch

Training code for Gaus-MVSNet, the unsupervised multi-view stereo network that
supplies depth to the GausSeg pipeline. The inference pipeline is on the
`gaus-mvsnet` branch.

## Installation

The devcontainer installs everything needed:

```bash
cd .devcontainer
docker compose build
docker compose up -d
docker compose exec gausseg bash
```

<details>
<summary><b>Installing without Docker</b></summary>

- `torch==2.4.0` and `torchvision==0.19.0` (cu124)
- gsplat 1.5.3 from its prebuilt pt24cu124 wheel, with its dependencies from PyPI:

  ```bash
  pip download --no-deps gsplat==1.5.3 -d /tmp/gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
  pip install --no-deps /tmp/gsplat/gsplat-*.whl
  pip install jaxtyping rich ninja
  ```

- `numpy`, `opencv-python-headless`, `Pillow`, `pyyaml`, `tqdm`

</details>

## Data

The DTU training set in the [MVSNet](https://github.com/YoYo000/MVSNet)
preprocessing, plus the full-resolution depth maps (`Depths_raw`), placed at
the paths `data.dtu_root` and `data.dtu_depths_raw` in
[`config.yaml`](config.yaml) point to (symlinks are fine):

```
data/dtu/mvs_training/dtu/
├── Cameras/
│   ├── pair.txt
│   └── train/00000000_cam.txt ...
└── Rectified/
    └── scan1_train/rect_001_0_r5000.png ...
data/dtu/Depths_raw/
└── scan1/depth_map_0000.pfm, depth_visual_0000.png ...
```

## Training

```bash
python -m gaus_mvsnet.train --config config.yaml
```

[`config.yaml`](config.yaml) holds the configuration reported in the paper.
Checkpoints are written to `runs/gaus-mvsnet/model_XXXXXX.ckpt` after every
epoch, alongside a per-iteration `train_log.csv`.

## Using the checkpoint

Check out the `gaus-mvsnet` branch and point `gaus_mvsnet.checkpoint` in its
`config.yaml` at the last checkpoint, `model_000014.ckpt`.

## License

[MIT](LICENSE), except [`gaus_mvsnet/model/`](gaus_mvsnet/model/), which is
vendored from [CasMVSNet](https://github.com/alibaba/cascade-stereo) and
[RC-MVSNet](https://github.com/Boese0601/RC-MVSNet) and covered by
[`gaus_mvsnet/model/LICENSE`](gaus_mvsnet/model/LICENSE).
