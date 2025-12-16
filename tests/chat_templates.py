import mlx.core as mx
from mlx_raclate.utils.utils import load


tested_models = [
    "LiquidAI/LFM2-350M",
]

chat_template = """{{- bos_token -}}
{%- set system_prompt = "" -%}
{%- set ns = namespace(system_prompt="") -%}
{%- if messages[0]["role"] == "system" -%}
	{%- set ns.system_prompt = messages[0]["content"] -%}
	{%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
	{%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: <|tool_list_start|>[" -%}
	{%- for tool in tools -%}
		{%- if tool is not string -%}
            {%- set tool = tool | tojson -%}
		{%- endif -%}
		{%- set ns.system_prompt = ns.system_prompt + tool -%}
        {%- if not loop.last -%}
            {%- set ns.system_prompt = ns.system_prompt + ", " -%}
        {%- endif -%}
	{%- endfor -%}
	{%- set ns.system_prompt = ns.system_prompt + "]<|tool_list_end|>" -%}
{%- endif -%}
{%- if ns.system_prompt -%}
	{{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
	{{- "<|im_start|>" + message["role"] + "\n" -}}
	{%- set content = message["content"] -%}
	{%- if content is not string -%}
		{%- set content = content | tojson -%}
	{%- endif -%}
	{%- if message["role"] == "tool" -%}
		{%- set content = "<|tool_response_start|>" + content + "<|tool_response_end|>" -%}
	{%- endif -%}
	{{- content + "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
	{{- "<|im_start|>assistant\n" -}}
{%- endif -%}
"""

def main():
    # Load the model and tokenizer
    model, tokenizer = load(
        "LiquidAI/LFM2-350M", 
        pipeline="embeddings", # models trained for sentence similarity will automatically use the "sentence-transformers" pipeline
    ) 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

    # if not getattr(tokenizer, "chat_template", None):
    #         raise ValueError("The tokenizer does not support chat templates.")
    messages = [
        {"role": "user", "content": "test_prompt"},
        {"role": "assistant", "content": "test_response"}
    ]
    templated = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False)
    print("Chat template working:", templated)
    
    

if __name__ == "__main__":
    main()
