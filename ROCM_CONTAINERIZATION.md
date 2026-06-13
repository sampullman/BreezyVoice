# Modern ROCm Containerization Plan

## Goal

Run BreezyVoice inside a modern, AMD-aligned ROCm container so the entire runtime stack is versioned together instead of being assembled piecemeal on the host.

This is the most practical next step if the target is:

- cleaner reproducibility
- fewer host-level Python and ABI issues
- a higher chance of stable ROCm Torch behavior
- a better base for further GPU tuning

It is not guaranteed to produce a huge speedup by itself, but it is the best foundation for further AMD-specific optimization.

## Why containerize this

The current host ROCm path works for the main Torch inference path, but getting there required a lot of compatibility repair:

- ROCm host runtime and GPU access had to be installed separately
- the Python stack had to be rebuilt around ROCm Torch
- Torchaudio 2.11 changed audio I/O behavior
- `torchcodec` from PyPI was CUDA-linked and unusable here
- ONNX Runtime ROCm exposed providers but failed to load `libhipblas.so.2`
- the repo needed a few compatibility fixes for newer Torch/Torchaudio

Those are exactly the kinds of problems a tested ROCm container is meant to reduce.

AMD’s ROCm docs explicitly recommend their prebuilt PyTorch containers as a tested path for custom workloads.

## What “containerize” should mean here

Use a container that already contains:

- a ROCm-supported Ubuntu base
- a ROCm-supported Python version
- ROCm-enabled PyTorch
- a matching Triton/AOTriton package layout
- ROCm libraries already in the expected paths

Then layer the BreezyVoice repo and only the repo-specific Python dependencies on top.

The host should provide only:

- kernel / amdgpu / ROCm device access
- `/dev/kfd`
- `/dev/dri`
- user membership in `render` and `video`

The host should not be responsible for reconciling Python package ABI conflicts.

## Recommended container base

Start from an official ROCm PyTorch image, not from a generic Ubuntu or Python image.

Practical direction:

- use a ROCm 7.x Ubuntu 24.04 PyTorch image
- prefer an image line AMD documents for Ryzen / Radeon ROCm usage
- pin the exact image tag in the Dockerfile or compose config

Why:

- PyTorch, Triton, HIP, and user-space ROCm libs are already matched
- this avoids the “Torch works, Triton half-works, Torchaudio changed, ORT wheel mismatched” situation

## Recommended repo layout

Add a separate AMD-focused container path instead of replacing the current Docker setup.

Suggested files:

- `Dockerfile.rocm`
- `docker-compose.rocm.yml`
- optionally `requirements.rocm.txt`

Keep the existing CPU / generic path intact.

## Recommended runtime design inside the container

Use the container for the API and inference runtime only.

Keep these behaviors:

- mount the repo into the container during development
- mount a persistent Hugging Face cache
- mount output directories if needed
- keep model path configurable with `MODEL_PATH`
- keep runtime/device settings configurable through env vars

Useful mounts:

- repo source: `/workspace/BreezyVoice`
- HF cache: `/root/.cache/huggingface` or a non-root equivalent

## Recommended container run configuration

At minimum the container needs access to:

- `/dev/kfd`
- `/dev/dri`
- group permissions equivalent to `video` and `render`
- larger shared memory than the Docker default

Typical Docker flags / compose equivalents:

- `--device=/dev/kfd`
- `--device=/dev/dri`
- `--group-add video`
- `--group-add render`
- `--ipc=host` or a larger `shm_size`

If using compose, make those explicit in `docker-compose.rocm.yml`.

## What should stay on CPU vs GPU

Based on current measurements:

- the main Torch model path is the only meaningful GPU target
- the ONNX frontend path is small enough that it is not the main lever

That means the first container target should be:

- Torch on ROCm for `llm`, `flow`, `hift`
- keep frontend ONNX on CPU unless MIGraphX can be made to work cleanly

This is important because the frontend preprocess path was already under one second in testing.

## Why not prioritize ONNX Runtime ROCm in the container first

ONNX Runtime’s ROCm Execution Provider is not the long-term AMD path anymore. ONNX Runtime’s own docs say:

- ROCm EP was removed starting with 1.23
- ROCm 7.0 was the last officially AMD-supported ROCm EP release
- users should migrate to MIGraphX EP

On this machine, `onnxruntime-rocm` exposed ROCm providers but failed to load due to a `libhipblas.so.2` dependency mismatch against ROCm 7.2 host libraries.

Even if fixed, the likely gain is small because frontend time is small.

## Recommended dependency strategy in the container

Do not install the repo’s current `requirements.txt` blindly into the ROCm image.

Reasons:

- it mixes training and inference dependencies
- some pins are old and Python-version-sensitive
- some packages are irrelevant to local API inference
- some packages trigger avoidable source builds on Python 3.12

Instead:

1. start from the ROCm image’s built-in Torch stack
2. install only the runtime dependencies needed by:
   - `api.py`
   - `single_inference.py`
   - `cosyvoice/cli/*`
   - `cosyvoice/utils/*`
3. keep AMD-specific overrides in a separate requirements file

Suggested approach:

- `requirements.runtime.txt`
- `requirements.rocm-runtime.txt`

## Recommended AMD container experiments

Inside the container, compare these configurations:

1. CPU baseline
2. ROCm Torch, standard path
3. ROCm Torch with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
4. ROCm Torch with different thread settings
5. optional MIGraphX frontend experiment

Keep the same request text and API timing method for all comparisons.

## What we already know from host testing

On this machine:

- CPU in the modern ROCm env:
  - startup about `6.96s`
  - preprocess about `0.66s`
  - inference about `24.77s`
  - total about `25.43s`

- ROCm Torch without experimental AOTriton:
  - first request slower than CPU
  - warm request only slightly better than CPU

- ROCm Torch with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`:
  - startup about `6.91s`
  - preprocess about `0.73s`
  - inference about `16.04s` then `17.28s`
  - total about `16.78s` then `18.01s`

So the container project should be judged against that `~16.8s` best result, not against the older minute-scale baseline.

## What a successful container outcome looks like

Minimum success:

- same or better performance than the current host ROCm setup
- simpler setup
- fewer compatibility workarounds
- reproducible launch command

Strong success:

- consistent `~16-18s` or better for the short test prompt
- fewer warnings and fewer fragile Python package interactions
- cleaner path to further GPU experiments

## What would count as failure

Containerization is not worth adopting if:

- it performs materially worse than the current host ROCm result
- it still requires many host Python workarounds
- it still leaves Torch / Triton / ORT mismatched

In that case the container should be treated as a dev sandbox, not the default deployment path.

## Recommended first implementation

1. Create `Dockerfile.rocm` from an official ROCm PyTorch base image.
2. Add only BreezyVoice runtime dependencies, not the full repo requirements.
3. Mount the repo and Hugging Face cache.
4. Expose `/dev/kfd` and `/dev/dri`.
5. Launch the existing API with:
   - `BREEZYVOICE_DEVICE=rocm`
   - `BREEZYVOICE_ORT_DEVICE=cpu`
   - `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
6. Benchmark the same short request.

## Recommended default env inside the container

Start with:

```bash
export BREEZYVOICE_DEVICE=rocm
export BREEZYVOICE_ORT_DEVICE=cpu
export BREEZYVOICE_USE_TTSFRD=0
export BREEZYVOICE_NUM_THREADS=16
export BREEZYVOICE_ORT_INTRA_OP_THREADS=8
export BREEZYVOICE_ORT_INTER_OP_THREADS=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

## Main risks

- official ROCm container image version may still need tuning for this APU
- Torchaudio / TorchCodec behavior may still need repo-side audio I/O bypasses
- MIGraphX / ORT acceleration may still be fragile or low-value
- GPU results may vary noticeably between cold and warm runs

## Recommendation

If the goal is a maintainable AMD deployment, containerization is worth doing.

If the goal is only maximum raw speed, containerization is still worth doing first because it is the cleanest way to test newer ROCm/PyTorch combinations without destabilizing the host.

## References

- AMD Ryzen / Radeon ROCm install docs:
  - https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html
- AMD Radeon ROCm docs:
  - https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html
- ONNX Runtime ROCm EP docs:
  - https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html
- ONNX Runtime MIGraphX EP docs:
  - https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
