"""Serve a HuggingFace model with vLLM on Modal.

Usage:
    # Deploy and get the endpoint URL
    modal deploy scripts/serve_vllm_modal.py

    # Or run ephemerally (stops when you ctrl-c)
    modal run scripts/serve_vllm_modal.py

The endpoint URL will be printed to stdout. Use it as the api_base in mini-swe-agent config.

Environment variables:
    MODEL_ID: HuggingFace model ID (default: open-thoughts/OpenThinker-Agent-v1-SFT)
    GPU_COUNT: Number of H100 GPUs (default: 2)
"""

import os
import subprocess

import modal

MODEL_ID = os.environ.get("MODEL_ID", "open-thoughts/OpenThinker-Agent-v1-SFT")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))

MINUTES = 60
VLLM_PORT = 8000

app = modal.App("vllm-serve")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=f"H100:{GPU_COUNT}",
    scaledown_window=20 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Launch vLLM's OpenAI-compatible server with tool call support."""
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        MODEL_ID,
        "--tensor-parallel-size",
        str(GPU_COUNT),
        "--max-model-len",
        "32768",
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test():
    """Quick smoke test: hit /health then /v1/models."""
    import aiohttp

    url = await serve.get_web_url.aio()
    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Health check: {url}/health")
        async with session.get("/health", timeout=aiohttp.ClientTimeout(total=9 * MINUTES)) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed")

        async with session.get("/v1/models") as resp:
            import json
            data = await resp.json()
            print(json.dumps(data, indent=2))
