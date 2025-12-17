import json
from llm import LLMEngine, SYSTEM_PROMPT, TOOLS


engine = LLMEngine("Qwen/Qwen2.5-VL-3B-Instruct")

def run_query(user_query: str, max_new_tokens=256, max_steps=10):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_query}]}
    ]

    plan = None
    completed_steps = []
    phase = "plan"

    for step in range(max_steps):

        # PHASE 1 - PLAN
        if phase == "plan":

            llm_output = engine.generate(messages, max_new_tokens)

            # JSON-only
            try:
                response = json.loads(llm_output)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Step {step}: model returned non-JSON output:\n{llm_output}"
                )

            response_phase = response.get("phase")
            if response_phase != "plan":
                raise RuntimeError(
                    f"Step {step}: expected phase 'plan', got '{response_phase}'"
                )

            plan = response.get("plan")
            if not isinstance(plan, list) or not plan:
                raise ValueError("Plan must be a non-empty list")

            # Validate plan tools
            invalid = [t for t in plan if t not in TOOLS]
            if invalid:
                raise RuntimeError(
                    f"Plan contains invalid tools: {invalid}. "
                    f"Allowed: {list(TOOLS.keys())}"
                )

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": llm_output}]
            })

            phase = "tool"
            continue

        # PHASE 2 - TOOL EXECUTION
        if phase == "tool":

            # Force next tool
            if len(completed_steps) < len(plan):
                expected_tool = plan[len(completed_steps)]

                messages.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "NEXT STEP (MANDATORY):\n"
                            f"You MUST call the tool '{expected_tool}' now.\n"
                            "You are FORBIDDEN from answering with phase='final'.\n"
                            "Respond ONLY with valid JSON:\n"
                            f'{{"phase":"tool","tool":"{expected_tool}","arguments":{{...}}}}'
                        )
                    }]
                })

            llm_output = engine.generate(messages, max_new_tokens)

            try:
                response = json.loads(llm_output)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Step {step}: model returned non-JSON output:\n{llm_output}"
                )

            response_phase = response.get("phase")

            # Tool call
            if response_phase == "tool":
                tool_name = response.get("tool")

                expected_tool = plan[len(completed_steps)]
                if tool_name != expected_tool:
                    raise RuntimeError(
                        f"Expected tool '{expected_tool}', got '{tool_name}'"
                    )

                args = response.get("arguments", {})
                result = TOOLS[tool_name](**args)

                completed_steps.append(tool_name)

                # Assistant message
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": llm_output}]
                })

                # Tool result message
                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": json.dumps(result)
                })

                if len(completed_steps) == len(plan):
                    phase = "final"

                continue

            # PHASE 3 - FINAL
            if response_phase == "final":
                if len(completed_steps) != len(plan):
                    messages.append({
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": (
                                "ERROR: You attempted to answer before completing "
                                "all required tool calls.\n"
                                "You MUST continue tool execution."
                            )
                        }]
                    })
                    continue

                return response.get("answer", "").strip()

            raise RuntimeError(
                f"Step {step}: invalid phase '{response_phase}' in tool phase"
            )

    raise RuntimeError("Max steps reached without final answer")

