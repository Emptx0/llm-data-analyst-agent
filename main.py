import json
from llm import LLMEngine, SYSTEM_PROMPT, TOOLS


engine = LLMEngine("Qwen/Qwen2.5-VL-3B-Instruct")

def run_query(user_query: str, max_new_tokens=128):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_query}]}
    ]
    
    llm_output = engine.generate(messages, max_new_tokens)
    
    tool_call = json.loads(llm_output)
    
    tool_name = tool_call["tool"]
    args = tool_call["arguments"]

    result = TOOLS[tool_name](**args)
    print("TOOL RESULT:", result)

