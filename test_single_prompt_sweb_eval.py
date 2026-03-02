import json, yaml, requests
from datasets import load_dataset
from jinja2 import StrictUndefined, Template

with open('src/minisweagent/config/benchmarks/swebench.yaml') as f:
    config = yaml.safe_load(f)

ds = load_dataset('SWE-bench/SWE-bench_Multilingual', split='test')
inst = [x for x in ds if x['instance_id'] == 'sharkdp__bat-562'][0]

system_msg = config['agent']['system_template'].strip()
user_msg = Template(config['agent']['instance_template'], undefined=StrictUndefined).render(task=inst['problem_statement'])

tool_def = json.dumps({"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}})

# Qwen3 chat template with tools
prompt = f"""<|im_start|>system
{system_msg}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_def}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""

print(f"Prompt length: {len(prompt)} chars")

resp = requests.post(
    "https://cuv1sqsh8nzhdf.r431.modal.host/v1/completions",
    headers={"Authorization": "Bearer swesmith", "Content-Type": "application/json"},
    json={
        # "model": "AlienKevin/swe-smith-rs-base-qwen3-8b-teacher-glm-4.6",
        # "model": "AlienKevin/swe-smith-rs-base-qwen3-8b-teacher-gpt-5-mini",
        # "model": "AlienKevin/swe-smith-rs-base-qwen3-8b-teacher-minimax-m2.5",
        # "model": "qwen3-8b-sft-epoch0",
        # "model": "Qwen/Qwen3-8B",
        "model": "swe-smith-rs-glm-4.6-hermes",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.0,
    },
    timeout=120,
)
data = resp.json()
text = data['choices'][0]['text']
finish = data['choices'][0]['finish_reason']

print(f"\nfinish_reason: {finish}")
print(f"output length: {len(text)} chars")
print(f"\n=== RAW OUTPUT ===")
print(text)
print(f"\n=== REPR ===")
print(repr(text))
