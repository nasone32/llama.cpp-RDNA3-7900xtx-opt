# llama.cpp RDNA3 / RX 7900 XTX build

This is the llama.cpp build I use for Qwen on AMD RDNA3.

It is a concoction of PRs, some AMD patches originally written for RDNA3.5 but compatible with RDNA3, experimental llama.cpp work that has not landed upstream yet, the RDNA Boost patch series, and a few changes made specifically for this setup.

So far it has been tested on Linux with ROCm using:

- one RX 7900 XTX;
- two RX 7900 XTX cards over PCIe;
- Qwen3.8 27B dense `Q8_0` with MTP;
- Qwen3.8 Flash Next MoE `UD_Q3_XXL` with MTP.

The main goal is simple: get as much useful performance as possible out of RDNA3 without pretending PCIe is Infinity Fabric.

This is experimental. It works well here, but expect sharp edges and do your own testing.

## Performance

Best prompt-processing results measured on the two-card RX 7900 XTX system:

- **Qwen3.8 Flash Next MoE `UD_Q3_XXL`:** about **920 tokens/s**;
- **Qwen3.8 27B dense `Q8_0`:** about **1,600 tokens/s** with tensor parallel, even though communication is limited by PCIe.

These are measured results, not guaranteed numbers. Model quantization, prompt length, context size, ROCm version, PCIe layout, clocks, and background GPU use all matter.

## What is different from upstream llama.cpp

- RDNA3 and RDNA3.5 kernel tuning for flash attention, TOP_K, MMQ, MMVQ, GDN, and routed MoE workloads.
- Internal two-GPU HIP AllReduce and peer-to-peer transfers, without RCCL.
- Optional Q8_0 between cards for inter-GPU AllReduce. This cuts PCIe traffic.
- Fused AllReduce plus residual add, which avoids another full pass over the activation.
- Faster Qwen3.8 Flash prefill through "direct lazy loading" of Ngram tables, and the "chunked BF16 gated-delta-net".
- Adaptive MTP draft depth.
- Experimental MoE expert cache, with additional work to remove overhead from the original implementation.

## Extra options in this build

### Adaptive MTP

```sh
--spec-type draft-mtp-adaptive
--spec-draft-n-max 3
```

Instead of always drafting a fixed number of MTP tokens, adaptive MTP changes the draft depth at runtime, up to `--spec-draft-n-max`. If deeper drafts are being accepted it can use them; when acceptance drops it backs off instead of wasting work.

### Direct lazy loading

```sh
--load-mode none
--lazy-mode on-direct
```

This combination makes a big difference to prompt processing on Qwen3.8 Flash. It avoids the slower default loading path and lets the lazy PLE path read directly where possible.

### MoE expert cache

```sh
--moe-expert-cache 64
--moe-expert-cache-inserts 2
```

The cache keeps recently used, host-offloaded MoE experts in VRAM. This can improve decode when system RAM bandwidth is the bottleneck.

I personally run with `--moe-expert-cache 0` because on my workload the prompt-processing slows down to a crawl and I find it unusable. BUT The feature is still included, and this fork removes a good chunk of overhead from the original PR (i might push it there also).

Cache size is hardware and model dependent. Start low, watch VRAM usage, and increase it. more cached experts = more speed. with 288 expert you get a 95% hit rate so things can decode fast, i've seen 45tk/s with MTP, but decode isn't everything

## Applied series

- `6ed7fb04f` MoE expert cache: GPU-resident LRU cache for host-offloaded expert weights
- `d7f316ec4` qwen4exp: use channels-major SSM conv for GDN
- `33611a98a` ssm_conv: add channels-major input mode; drop delta-net transpose
- `e2188eb26` delta-net: force contiguous conv-state concat input
- `ed11a0d2f` ggml-cuda: tune D=256 tile flash-attn config for RDNA3.5
- `670512936` ggml-cuda: add dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K
- `9a764c613` ggml-cuda: size MMQ tile to MoE tokens-per-expert
- `7dfa528e3` mmq: opt-in compacted MoE tiling for RDNA3.5
- `7f1d25f7e` ROCm: make TOP_K wave32-native
- `10fdba9ae` rename kernels/functions
- `7f3e1e4d0` ROCm: add hybrid TOP_K kernels
- `d2d89512d` qwen4exp: gather-based sparse attention for QSA decode (#28213)
- `abca85cdc` qwen4exp: direct reads for lazy PLE tables (#28136)
- `47ff3777a` CUDA: size routed MoE MMQ N-tiles from typical expert width (#24546)
- `84b31d4cd` llama : add backends of the other model to the context
- `22ed83e9a` fix DFlash2 with tensor split (assisted by DeepSeek v4 Flash)
- `e06dcf630` ggml: add Q8 wire, residual fusion, and AllReduce tracing
- `7c5bb5cb9` ggml: enable internal HIP all-reduce and P2P
- `cec239cb8` rdna-boosts: block 13: fused MoE gate+up+GLU MMQ + mmvq short-K item-split
- `b7b53d5bf` rdna-boosts: block 11: skip CUDA graphs for multi-token PRE-FILL
- `efa4e8641` rdna-boosts: block 10: k-quant-boosts - Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR
- `91de35039` rdna-boosts: block 09: meta-buffer compute-container headroom for
- `92426ebba` rdna-boosts: block 08: fused-core prefill kernels and GPU bit-identical
- `c7bb928fe` rdna-boosts: block 07: meta device-wrapper skip
- `20a3f9116` rdna-boosts: block 06: host-buffer revert for discrete GPUs
- `c92cb00c3` rdna-boosts: block 05: CPU bit-identical decode/verify batches
- `371270e17` rdna-boosts: block 04: RDNA4 WMMA flash-attn + Q6_K mmq prefill perf
- `9b7126690` rdna-boosts: block 03: BF16 KV cache and native-BF16 flash-attn
- `4169fbbf5` rdna-boosts: block 02: fused chunked gated-delta-net prefill kernel
- `10579a736` rdna-boosts: block 01: adaptive MTP draft depth

## Build

Install a working ROCm toolchain plus Git, CMake, a C/C++ compiler, and the usual build tools. AMD's installation guide is available [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

Clone the repository:

```sh
git clone https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt.git
cd llama.cpp-RDNA3-7900xtx-opt
```

Build for the RX 7900 XTX (`gfx1100`):

```sh
cmake -S . -B build-rocm-gfx1100-portable \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang \
    -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DLLAMA_BUILD_TESTS=ON

cmake --build build-rocm-gfx1100-portable --config Release -j "$(nproc)"
```

The binaries will be in `build-rocm-gfx1100-portable/bin/`.

These paths match the ROCm 7.14 layout used for the tested build. If your ROCm installation lives somewhere else, change all three compiler paths to match it.

`GGML_HIP_ROCTX` is intentionally not enabled. It only adds profiling markers, requires `rocprofiler-sdk-roctx`, and is not needed for normal inference.

OpenSSL is optional. Without its development headers the build still works, but llama.cpp's built-in HTTPS support is disabled.

For another GPU, check its architecture with `rocminfo` and replace `gfx1100`. This README only documents the hardware that has actually been tested.

## Runtime environment

With the ROCm 7.14 layout above, expose its runtime libraries before starting the binaries:

```sh
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/llvm/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Without this line the loader may fail with errors such as `libhipblas.so.3: cannot open shared object file`.

Now check the device list:

```sh
./build-rocm-gfx1100-portable/bin/llama-server --list-devices
```

If ROCm also sees an integrated GPU, keep only the two discrete cards visible. Adjust the IDs to match your machine:

```sh
export HIP_VISIBLE_DEVICES=0,1
```

For the two-card tensor-parallel setup:

```sh
export GGML_CUDA_ALLREDUCE=internal
export GGML_CUDA_AR_WIRE=q8_0
export GGML_CUDA_AR_Q8_THRESHOLD=1048576
export GGML_CUDA_AR_FUSED_RESIDUAL=1
export GGML_CUDA_GDN_CHUNKED_BF16=1
export HIP_VISIBLE_DEVICES=0,1
```

Yes, these variables say `CUDA` even on AMD. That is the shared backend naming used by llama.cpp.

- `GGML_CUDA_ALLREDUCE=internal` selects this fork's internal two-GPU AllReduce instead of RCCL.
- `GGML_CUDA_AR_WIRE=q8_0` quantizes large inter-GPU AllReduce payloads to Q8_0, reducing PCIe traffic.
- `GGML_CUDA_AR_Q8_THRESHOLD=1048576` keeps tensors smaller than 1 MiB unquantized, because for small transfers conversion overhead hurts more than it helps.
- `GGML_CUDA_AR_FUSED_RESIDUAL=1` combines AllReduce and the following residual add, avoiding a separate read/write pass.
- `GGML_CUDA_GDN_CHUNKED_BF16=1` explicitly keeps the fast chunked BF16/WMMA gated-delta-net prefill path enabled on RDNA3. It is already the default for supported shapes in this build. It is near-lossless, not bit-exact.
- `HIP_VISIBLE_DEVICES=0,1` hides unwanted devices such as an integrated GPU. Run `--list-devices` first and use the correct IDs for your system.

For one GPU, you do not need the AllReduce variables. `GGML_CUDA_GDN_CHUNKED_BF16=1` is still useful for supported Qwen models.

## Example: Qwen3.8 27B dense with MTP

This configuration is for **two RX 7900 XTX cards**, with the desktop/monitor attached to `ROCm1`. Replace every `/path/to/...` value with the real path on your machine.

For the dense model, enable direct P2P and let `ROCm1` issue the transfers:

```sh
export GGML_CUDA_AR_P2P=1
export GGML_CUDA_AR_P2P_ISSUER=1
```

- `GGML_CUDA_AR_P2P=1` uses direct peer-to-peer GPU transfers when the PCIe topology supports them.
- `GGML_CUDA_AR_P2P_ISSUER=1` makes the second visible GPU issue those transfers. This matches the tested machine; try the other issuer if your PCIe topology is different.

```sh
/path/to/llama.cpp-RDNA3-7900xtx-opt/build-rocm-gfx1100-portable/bin/llama-server \
    --model /path/to/qwen3.8-27b-model.gguf \
    --alias qwen3.8-27b \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 50144 \
    --spec-type draft-mtp-adaptive \
    --spec-draft-n-max 3 \
    --device-draft ROCm0 \
    --split-mode tensor \
    --flash-attn on \
    --batch-size 4096 \
    --ubatch-size 1024 \
    --moe-expert-cache 0 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --presence-penalty 0.0 \
    --reasoning on \
    --reasoning-effort medium \
    --parallel 1 \
    --fit on \
    --fit-target 2800,2048 \
    --load-mode none \
    --lazy-mode on-direct
```

The Qwen3.8 27B `Q8_0` GGUF tested here already contains its MTP weights, so it does not need a separate `--model-draft` file. `--fit-target 2800,2048` deliberately leaves more headroom on `ROCm0` for MTP. The display is on `ROCm1` in this setup, so your own memory targets may need to be different.

## Example: Qwen3.8 Flash Next MoE with MTP

This configuration is also for **two RX 7900 XTX cards**, with the desktop/monitor attached to `ROCm1`.

For this partially offloaded MoE model, host staging was faster during prompt processing than direct P2P:

```sh
export GGML_CUDA_AR_P2P=0
```

```sh
/path/to/llama.cpp-RDNA3-7900xtx-opt/build-rocm-gfx1100-portable/bin/llama-server \
    --model /path/to/qwen3.8-flash-next-UD_Q3_XXL-model.gguf \
    --model-draft /path/to/qwen3.8-flash-next-mtp-model.gguf \
    --alias qwen3.8-flash-next-q3_k_xl \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 50144 \
    --spec-type draft-mtp-adaptive \
    --spec-draft-n-max 3 \
    --device-draft ROCm0 \
    --split-mode layer \
    --flash-attn on \
    --batch-size 4096 \
    --ubatch-size 1024 \
    --moe-expert-cache 0 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --presence-penalty 0.0 \
    --reasoning on \
    --reasoning-effort medium \
    --parallel 1 \
    --fit on \
    --fit-target 2800,2048 \
    --load-mode none \
    --lazy-mode on-direct
```

`--fit-target 2800,2048` leaves extra room on `ROCm0` for the MTP model. The MoE expert cache is disabled in this example on purpose. Enable it only after you have a stable baseline and enough free VRAM.

## Example: Qwen3.8 Flash Next MoE with MTP and expert cache

This is the high-context expert-cache configuration tested on **two RX 7900 XTX cards**. It offloads the MoE expert weights to system RAM and keeps the most recently used experts in a GPU-resident LRU cache.

Start with 288 slots and two uploads per layer and decode step:

```sh
MOE_CACHE_SLOTS="${MOE_CACHE_SLOTS:-288}"
MOE_CACHE_INSERTS="${MOE_CACHE_INSERTS:-2}"

/path/to/llama.cpp-RDNA3-7900xtx-opt/build-rocm-gfx1100-portable/bin/llama-server \
    --model /path/to/qwen3.8-flash-next-model.gguf \
    --model-draft /path/to/qwen3.8-flash-next-mtp-model.gguf \
    --alias qwen3.8-flash-next-q3_k_xl \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 150000 \
    --split-mode layer \
    --flash-attn on \
    --batch-size 2048 \
    --ubatch-size 64 \
    --spec-type draft-mtp-adaptive \
    --spec-draft-n-max 3 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    --presence-penalty 0.0 \
    --reasoning on \
    --reasoning-effort medium \
    --parallel 1 \
    --fit on \
    --fit-target 11000,10240 \
    --load-mode none \
    --device-draft ROCm0 \
    --lazy-mode on-direct \
    --seed 1234 \
    -ot "ffn_(gate|up|down)_exps\.weight=CPU,per_layer_token_embd\.weight=CPU" \
    --moe-expert-cache "$MOE_CACHE_SLOTS" \
    --moe-expert-cache-inserts "$MOE_CACHE_INSERTS"
```

The 288-slot default is the largest comfortably tested value at this context size on the reference machine. `--ctx-size 150000` is internally rounded to 150016. Expert-cache memory is not fully accounted for by `--fit`, so do not blindly increase the slot count: watch ROCm0 VRAM during startup and leave some headroom for the MTP context and compute buffers.

This mode improves decode when host RAM bandwidth is the bottleneck, but prompt processing is slower because the experts are offloaded. Use the non-cache MoE example above when prompt-processing speed matters more.

## Credits

This fork is based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and combines work from upstream llama.cpp contributors, AMD ROCm contributors, experimental pull requests, and the RDNA Boost series. Check the commit history for authorship and the exact changes carried by this branch.
