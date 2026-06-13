# ROCm Docker Setup

Use the ROCm container path only.

## Build

```bash
docker compose -f docker-compose.rocm.yml build
```

## Run

This is the latest efficient default from the measured runs:

```bash
BREEZYVOICE_FLOW_AMP=bfloat16 \
BREEZYVOICE_HIFT_AMP=bfloat16 \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
docker compose -f docker-compose.rocm.yml up -d
```

It keeps:

- prompt caching enabled
- ONNX frontend on CPU
- HiFT weight norm removed at inference load

## Check

```bash
docker logs -f breezyvoice-app-1
```

Look for:

- `flow_amp: 'bfloat16'`
- `hift_amp: 'bfloat16'`
- `hift_weight_norm_removed: True`

## Stop and clean up

```bash
docker compose -f docker-compose.rocm.yml down -v --remove-orphans
docker image rm breezyvoice:rocm
```
