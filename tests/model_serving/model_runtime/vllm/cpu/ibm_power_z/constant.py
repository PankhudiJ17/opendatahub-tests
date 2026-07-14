from typing import Any

FALCON3_7B_INSTRUCT_MODEL_PATH: str = "models/Falcon3-7B-Instruct"
LLAMA_3_2_1B_INSTRUCT_MODEL_PATH: str = "models/llama-32-1b-instruct"
PHI_4_MODEL_PATH: str = "models/phi-4"
MISTRAL_7B_INSTRUCT_MODEL_PATH: str = "models/Mistral-7B-v0.3"
GRANITE_3_1_8B_INSTRUCT_MODEL_PATH: str = "models/granite-3.1-8b-instruct"
ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT_MODEL_PATH: str = "models/ELYZA-japanese-Llama-2-7b-instruct"
GRANITE_3_3_8B_INSTRUCT_MODEL_PATH: str = "models/granite-3.3-8b-instruct"
DEEPSEEK_R1_DISTILL_LLAMA_8B_MODEL_PATH: str = "models/deepseek-r1-distill-llama-8b"
PHI_3_MINI_4K_INSTRUCT_MODEL_PATH: str = "models/Phi-3-mini-4k-instruct"
META_LLAMA_3_1_8B_INSTRUCT_MODEL_PATH: str = "models/Meta-Llama-3.1-8B-Instruct"
TINYLLAMA_1_1B_CHAT_V1_0_MODEL_PATH: str = "models/tinyllama-1.1b-chat-v1.0"
MINISTRAL_3B_INSTRUCT_MODEL_PATH: str = "models/ministral-3b-instruct"
GRANITE_3_1_2B_INSTRUCT_MODEL_PATH: str = "models/granite-3.1-2b-instruct"
LLAMA_3_2_3B_INSTRUCT_MODEL_PATH: str = "models/llama-32-3b-instruct"
GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH: str = "models/granite-3b-code-instruct-2k"


# Environment variables for vLLM CPU on Power/Z architectures
# VLLM_CPU_KVCACHE_SPACE limits KV cache memory to prevent OOM
IBM_POWER_Z_ENV_VARIABLES: list[dict[str, str]] = [
    {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "8"},
    {"name": "VLLM_WORKER_MULTIPROC_METHOD", "value": "spawn"},
    {"name": "OMP_NUM_THREADS", "value": "16"},
]

IBM_POWER_Z_PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "16", "memory": "70Gi"},
    "limits": {"cpu": "16", "memory": "70Gi"},
}

IBM_POWER_Z_SERVING_ARGUMENT: list[str] = [
    "--dtype=bfloat16",
    "--model=/mnt/models",
    "--max-model-len=2048",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=256",
    "--uvicorn-log-level=debug",
]

IBM_POWER_Z_CHAT_INFERENCE_REQUEST: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 50,
}

