import json
from llm import LLMEngine, SYSTEM_PROMPT, TOOLS


engine = LLMEngine("Qwen/Qwen2.5-VL-3B-Instruct")

def run_query(user_query: str, max_new_tokens=256):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_query}]}
    ]

    tool_results = []

    for _ in range(5):  # max tool steps
        llm_output = engine.generate(messages, max_new_tokens)

        # проба парсингу як tool call
        try:
            tool_call = json.loads(llm_output)

            tool_name = tool_call["tool"]
            args = tool_call.get("arguments", {})

            result = TOOLS[tool_name](**args)

            # зберігаємо результат
            tool_results.append({
                "tool": tool_name,
                "result": result
            })

            # повертаємо результат LLM як context
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": llm_output}]
            })
            messages.append({
                "role": "tool",
                "content": [{"type": "text", "text": json.dumps(result)}]
            })
            
            continue

        except json.JSONDecodeError:
            last_text_response = llm_output
            
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": llm_output}]
            })
            
            continue

    return last_text_response or "Could not complete analysis."

