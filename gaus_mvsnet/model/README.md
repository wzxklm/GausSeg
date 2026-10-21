# Vendored backbone

`casmvsnet.py` and `modules.py` hold the cascade MVS backbone Gaus-MVSNet
builds on: a shared FPN feature extractor, variance-fused cost volumes, and
3D encoder–decoder regularization over three cascade stages. The code comes
from CasMVSNet (Gu et al., CVPR 2020,
[alibaba/cascade-stereo](https://github.com/alibaba/cascade-stereo)) in the
form used by RC-MVSNet (Chang et al., ECCV 2022,
[Boese0601/RC-MVSNet](https://github.com/Boese0601/RC-MVSNet)).

`casmvsnet.py` and `modules.py` are kept close to that upstream so checkpoints
the trainer writes load into inference with `strict=True` — parameter names are
part of the interface here. Only three edits were made, none of which touch
module structure or parameter names:

- the two constructor `print()` calls and one in `FeatureNet` were removed
- `sys.path.append("..")` was dropped from `modules.py`
- both files were moved under `gaus_mvsnet/model/`

`warping.py` holds the inverse warping and masked SSIM the photometric loss is
built on, from RC-MVSNet's `losses/homography.py` and `losses/modules.py`. It
has no parameters, and differs from upstream in two respects:

- the sampling mask tests `y1 <= max_y` instead of `y0 <= max_y`; the latter
  admits the bottom row, whose lower neighbour is clamped back onto itself
- unused imports, locals and commented-out code were removed

CasMVSNet and RC-MVSNet are MIT-licensed, as is this repository, but their
notices travel with the code: see [`LICENSE`](LICENSE) in this directory, which
covers these three files instead of the repository's own.

The rest of `gaus_mvsnet/` — the training backbone wrapper, the DTU dataset,
the photometric loss, the 3D Gaussian Splatting branch, and the trainer — is
this repository's own code and follows its conventions rather than upstream's.
