# llama.cpp MoE Expert Cache

Experimental llama.cpp fork with a GPU-resident LRU cache for MoE experts that are otherwise offloaded to system RAM.

The cache is intended to reduce memory bandwidth pressure during token generation when a large MoE model does not fit entirely in VRAM. This version is currently tested on Linux with ROCm and RDNA 3 GPUs.

## Build with ROCm

Install the standard build tools and a working ROCm toolchain first. See the [ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) if ROCm is not already installed.

```sh
git clone https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt.git
cd llama.cpp-RDNA3-7900xtx-opt

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build-rocm \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-rocm --config Release -j
```

Change `gfx1100` to the architecture reported for your GPU by `rocminfo`.

The binaries are created in `build-rocm/bin/`.

## Run

The expert cache works on MoE expert tensors offloaded to the CPU with `-ot`:

```sh
./build-rocm/bin/llama-server \
    --model /path/to/model.gguf \
    --flash-attn on \
    -ot "ffn_(gate|up|down)_exps\\.weight=CPU" \
    --moe-expert-cache 64 \
    --moe-expert-cache-inserts 2
```

- `--moe-expert-cache N` sets the number of GPU cache slots per host-resident MoE layer.
- `--moe-expert-cache-inserts N` limits expert uploads per layer and decode step.
- Start with a small cache and increase it while monitoring VRAM usage.
- Set `--moe-expert-cache 0` to disable the feature.

All regular llama.cpp options remain available. Run `llama-server --help` for the complete list.

## Status

This is experimental software. The current implementation is optimized and tested for the author's ROCm setup, but it still needs testing across more models and hardware.

Based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The original project documentation is available in [`docs/`](docs/).
