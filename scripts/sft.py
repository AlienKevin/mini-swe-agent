"""Fine-tune Qwen3-8B on SWE-smith-rs GLM-4.6 trajectories using TRL SFTTrainer on Modal.

Usage:
    modal run scripts/sft.py
"""

import json
import subprocess

import modal

MODEL_ID = "Qwen/Qwen3-8B"
DATASET_ID = "AlienKevin/SWE-smith-rs-glm-4.6-trajectories"

OUTPUT_VOLUME_PATH = "/output"
MODEL_CACHE_PATH = "/models"
OUTPUT_EXP_DIR = f"{OUTPUT_VOLUME_PATH}/exps/trl_qwen3_8b_swesmith_rs_glm_46"

GPU_COUNT = 8  # Max per Modal container; use grad_accum=2 for effective batch size of 16
GPU_TYPE = "H100"

app = modal.App("sft-qwen3-8b-swesmith-rs")

sft_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'trl[liger,peft]' wandb hf_transfer",
        "uv pip install --system --no-build-isolation flash-attn",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

output_volume = modal.Volume.from_name("sft-models", create_if_missing=True)
model_volume = modal.Volume.from_name("vllm-models", create_if_missing=True)

BASH_TOOL_SCHEMA = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command in the terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]
)

# Patched Qwen3 chat template with {% generation %}/{% endgeneration %} markers
# around assistant turns (including tool_calls) so that TRL's assistant_only_loss
# correctly masks non-assistant tokens from the loss.
# NOTE: This is a raw string to avoid any escaping issues.
QWEN3_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {% generation %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- content }}
            {%- endif %}
        {%- else %}
            {{- content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{\"name\": \"' }}
                {{- tool_call.name }}
                {{- '\", \"arguments\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
        {% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

# Training script — NOT an f-string to avoid Jinja/Python brace conflicts.
# Constants are injected by writing them as separate files at runtime.
TRAIN_SCRIPT = r"""
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


class EpochCheckpointCallback(TrainerCallback):
    # Save a named checkpoint (epoch0, epoch1, ...) and commit the volume.
    # Uses trainer.save_model() to go through the proper FSDP save path
    # (accelerator.get_state_dict), matching the format of the final save.

    def __init__(self):
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch) - 1
        epoch_dir = os.path.join(args.output_dir, f"epoch{epoch_idx}")
        self.trainer.save_model(epoch_dir)
        if state.is_world_process_zero:
            import modal
            modal.Volume.from_name("sft-models").commit()
            print(f"Saved checkpoint to {epoch_dir}")


def main():
    # Read constants written by the Modal launcher
    with open("/tmp/sft_config.json") as f:
        config = json.load(f)

    model_id = config["model_id"]
    dataset_id = config["dataset_id"]
    output_exp_dir = config["output_exp_dir"]
    bash_tool_schema = config["bash_tool_schema"]

    with open("/tmp/chat_template.jinja") as f:
        chat_template = f.read()

    dataset = load_dataset(dataset_id, split="train")

    # Add tools column for tool-calling SFT
    dataset = dataset.map(lambda x: {"tools": bash_tool_schema})

    # Keep only columns needed for SFT
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ("messages", "tools")]
    )

    # Patch tokenizer with chat template that has generation markers
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = chat_template

    training_args = SFTConfig(
        output_dir=output_exp_dir,
        report_to="wandb",
        run_name="trl_qwen3_8b_swesmith_rs_glm_46",
        learning_rate=4e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,  # 8 GPUs x 2 x 1 = effective batch size 16
        num_train_epochs=3.0,
        seed=42,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",  # We handle saving manually via the callback
        assistant_only_loss=True,
        max_length=32768,
        use_liger_kernel=True,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
    )

    epoch_cb = EpochCheckpointCallback()
    trainer = SFTTrainer(
        model=model_id,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[epoch_cb],
    )
    epoch_cb.trainer = trainer

    trainer.train()
    trainer.save_model(output_exp_dir)


if __name__ == "__main__":
    main()
"""


@app.function(
    image=sft_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        OUTPUT_VOLUME_PATH: output_volume,
        MODEL_CACHE_PATH: model_volume,
    },
    secrets=[modal.Secret.from_name("kevin-wandb-secret")],
    timeout=86400,  # 24 hours
)
def train():
    """Launch multi-GPU training via torchrun."""
    # Write config as JSON to avoid f-string/Jinja brace conflicts
    config = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "output_exp_dir": OUTPUT_EXP_DIR,
        "bash_tool_schema": BASH_TOOL_SCHEMA,
    }
    with open("/tmp/sft_config.json", "w") as f:
        json.dump(config, f)

    # Write chat template as a separate file
    with open("/tmp/chat_template.jinja", "w") as f:
        f.write(QWEN3_CHAT_TEMPLATE)

    # Write training script
    with open("/tmp/train_sft.py", "w") as f:
        f.write(TRAIN_SCRIPT)

    # Write accelerate config for FSDP (shards model + optimizer across GPUs)
    import yaml

    accelerate_config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "fsdp_config": {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": True,
        },
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "num_processes": GPU_COUNT,
    }
    with open("/tmp/accelerate_config.yaml", "w") as f:
        yaml.dump(accelerate_config, f)

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "/tmp/accelerate_config.yaml",
        "/tmp/train_sft.py",
    ]
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    output_volume.commit()
    print(f"Training complete. Model saved to {OUTPUT_EXP_DIR}")


@app.local_entrypoint()
def main():
    train.remote()
