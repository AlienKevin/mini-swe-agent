"""Verify that the patched Qwen3 chat template correctly masks loss to assistant turns only.

Loads one sample from the dataset, applies the patched chat template, and prints
a color-coded view showing which tokens are included in the loss (assistant turns
+ tool_calls) vs masked out (system, user, tool responses).

Usage:
    python scripts/verify_loss_mask.py
"""

import json

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
DATASET_ID = "AlienKevin/SWE-smith-rs-glm-4.6-trajectories"

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

# Same patched template as in sft.py
QWEN3_CHAT_TEMPLATE = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
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
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {% generation %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- content }}
            {%- endif %}
        {%- else %}
            {{- content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\\"name\\": \\"' }}
                {{- tool_call.name }}
                {{- '\\", \\"arguments\\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
        {% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""


GREEN = "\033[42m"  # Green background = included in loss
RED = "\033[41m"    # Red background = masked from loss
RESET = "\033[0m"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    print("Loading dataset...")
    dataset = load_dataset(DATASET_ID, split="train")
    sample = dataset[0]
    messages = sample["messages"]
    tools = json.loads(BASH_TOOL_SCHEMA)

    # Print message structure overview
    print(f"\n{'='*80}")
    print(f"Sample: {sample.get('instance_id', 'N/A')}")
    print(f"Total messages: {len(messages)}")
    for i, msg in enumerate(messages):
        role = msg["role"]
        has_tool_calls = bool(msg.get("tool_calls"))
        content_preview = (msg.get("content") or "")[:80].replace("\n", "\\n")
        extra = " [has tool_calls]" if has_tool_calls else ""
        print(f"  [{i:3d}] {role:12s}{extra:20s} {content_preview}...")
    print(f"{'='*80}\n")

    # Apply chat template with assistant mask
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )

    input_ids = result["input_ids"]
    assistant_mask = result["assistant_masks"]

    assert len(input_ids) == len(assistant_mask), (
        f"Length mismatch: {len(input_ids)} input_ids vs {len(assistant_mask)} mask values"
    )

    total_tokens = len(input_ids)
    trained_tokens = sum(assistant_mask)
    masked_tokens = total_tokens - trained_tokens

    print(f"Total tokens:   {total_tokens}")
    print(f"Trained tokens: {trained_tokens} ({100*trained_tokens/total_tokens:.1f}%)")
    print(f"Masked tokens:  {masked_tokens} ({100*masked_tokens/total_tokens:.1f}%)")
    print()

    # Decode and color-code each token
    print("Color-coded output (GREEN=trained, RED=masked):")
    print(f"{'='*80}")

    # Show in chunks for readability
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids]

    current_color = None
    for token_str, mask_val in zip(decoded_tokens, assistant_mask):
        color = GREEN if mask_val else RED
        if color != current_color:
            if current_color is not None:
                print(RESET, end="")
            print(color, end="")
            current_color = color
        print(token_str, end="")
    print(RESET)
    print(f"\n{'='*80}")

    # Verify correctness: check that specific patterns are in the right category
    print("\nVerification checks:")
    full_text = tokenizer.decode(input_ids)

    # Build trained vs masked text
    trained_text = "".join(t for t, m in zip(decoded_tokens, assistant_mask) if m)
    masked_text = "".join(t for t, m in zip(decoded_tokens, assistant_mask) if not m)

    checks = [
        ("System prompt masked",        "<|im_start|>system" in masked_text),
        ("User messages masked",        "<|im_start|>user" in masked_text),
        ("Tool responses masked",       "<tool_response>" in masked_text),
        ("Assistant content trained",   "<|im_end|>" in trained_text),
        ("Tool calls trained",          "<tool_call>" in trained_text),
    ]

    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {label}")

    print()
    if all_passed:
        print("All checks passed! Loss masks correctly cover assistant turns and tool calls.")
    else:
        print("WARNING: Some checks failed. Review the color-coded output above.")


if __name__ == "__main__":
    main()
