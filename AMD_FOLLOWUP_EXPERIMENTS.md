# AMD Follow-Up Experiments

## Current baseline

Short request:

- `你好，這是速度驗證。`
- API path `POST /v1/audio/speech`

Best host ROCm result so far:

- ROCm Torch enabled
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- total about `16.8-18.0s`

Working container default:

- image `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`
- prompt cache enabled
- ONNX frontend kept on CPU
- weight norm removed in HiFT at inference load

## Latest measured container results

Cached-prompt API path, same short request:

- baseline:
  - startup `19.709s`
  - preprocess `0.287s`
  - inference `24.169s`
  - total `24.457s`
  - `llm 10.106s / flow 5.262s / hift 8.799s`
- `BREEZYVOICE_HIFT_AMP=bfloat16`:
  - startup `22.136s`
  - preprocess `0.372s`
  - inference `24.117s`
  - total `24.489s`
  - `llm 10.110s / flow 5.302s / hift 8.704s`
- `BREEZYVOICE_FLOW_AMP=bfloat16` and `BREEZYVOICE_HIFT_AMP=bfloat16`:
  - startup `17.650s`
  - preprocess `0.375s`
  - inference `20.620s`
  - total `20.996s`
  - `llm 10.076s / flow 1.859s / hift 8.684s`

## Conclusions

- `hift` mixed precision alone is not useful here.
- `flow + hift` mixed precision is a real improvement, but the gain comes almost entirely from `flow`.
- `llm` stays near `10.1s`.
- `hift` stays near `8.7-8.8s`.
- container performance is still worse than the best host ROCm result.
- `hipblasLtMatmul` fallbacks and MIOpen workspace warnings still persist.

## Keep

- ROCm container line `7.2 / py3.10 / 2.9.1`
- cached-prompt API path
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- ONNX frontend on CPU
- selective AMP support for `flow` and `hift`

## Next step

Do deeper `hift` work, not more `hift` AMP tuning.

Most likely next targets:

- profile the convolution-heavy HiFT path
- test vocoder-side inference changes that reduce output cost without obvious quality loss
- only revisit `llm` after `hift`
