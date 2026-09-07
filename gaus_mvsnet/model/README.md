# Vendored backbone

`casmvsnet.py` and `modules.py` hold the cascade MVS backbone Gaus-MVSNet
builds on: a shared FPN feature extractor, variance-fused cost volumes, and
3D encoder–decoder regularization over three cascade stages. The code comes
from CasMVSNet (Gu et al., CVPR 2020,
[alibaba/cascade-stereo](https://github.com/alibaba/cascade-stereo)) in the
form used by RC-MVSNet (Chang et al., ECCV 2022,
[Boese0601/RC-MVSNet](https://github.com/Boese0601/RC-MVSNet)).

Both are MIT-licensed, as is this repository, but their notices travel with
the code: see [`LICENSE`](LICENSE) in this directory, which covers these two
files instead of the repository's own.

They are kept close to that upstream so the released checkpoint loads with
`strict=True` — parameter names are part of the interface here. Only three
edits were made, none of which touch module structure or parameter names:

- the two constructor `print()` calls and one in `FeatureNet` were removed
- `sys.path.append("..")` was dropped from `modules.py`
- both files were moved under `gaus_mvsnet/model/`

The rest of `gaus_mvsnet/` — the COLMAP front-end, the inference dataset,
fusion, and the stage itself — is this repository's own code and follows its
conventions rather than upstream's.

Inference runs the backbone alone. The 3D Gaussian Splatting auxiliary branch
that supervises training is not part of this release.
