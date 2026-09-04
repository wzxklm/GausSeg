# pointops (vendored)

Third-party CUDA extension, **not** part of the GausSeg contribution. It is
redistributed here under its own license — see [LICENSE](LICENSE) in this
directory, which governs everything under `pointops/` and supersedes the
repository-root MIT license for these files.

| | |
| --- | --- |
| Upstream | [Pointcept/Pointcept](https://github.com/Pointcept/Pointcept), `libs/pointops` |
| License | MIT — Copyright (c) 2023 Pointcept |
| Origin | Point Transformer V2 ([Wu et al., NeurIPS 2022](https://arxiv.org/abs/2210.05666)) |

## Why it is vendored

`qs_sam3d/segmentation.py` uses `pointops.knn_query` for the scene-self KNN
majority vote that smooths segment labels. Upstream ships pointops only as a
subdirectory of the full Pointcept training framework, which is not otherwise a
dependency of this pipeline, so the extension is copied in and built on its own.

## Local delta

The snapshot predates upstream's current revision (upstream has since
reformatted to double quotes and added an `if opt:` guard in `setup.py`). The
one deliberate change is in `setup.py`:

```python
extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2', '-ccbin', 'g++-10']}
```

`-ccbin g++-10` pins nvcc's host compiler to the same toolchain the devcontainer
builds COLMAP and OpenMVS with. Drop it if you build against a different one.

Kernel sources and the `functions/` wrappers are unmodified, and are kept in
**upstream style** (single quotes, chained asserts) so the directory can be
re-synced against Pointcept without a reformatting diff. Do not apply this
repository's own conventions here.

## Building

```bash
cd pointops
CXX=g++-10 pip install --no-build-isolation .
```

Requires a CUDA toolkit matching your PyTorch build. Set `TORCH_CUDA_ARCH_LIST`
to your GPU's compute capability (the devcontainer uses `8.9` for Ada).
