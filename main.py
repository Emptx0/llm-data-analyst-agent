import json
from llm import LLMEngine, SYSTEM_PROMPT, TOOLS

from logger import setup_logger
import time


engine = LLMEngine("Qwen/Qwen2.5-VL-3B-Instruct")


def plan_phase(
        messages: list[dict],
        max_new_tokens: int,
        step: int,
        phase: str
        ) -> tuple[str, list]:

    llm_output = engine.generate(messages, max_new_tokens)

    # JSON-only
    try:
        response = json.loads(llm_output)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Step {step}, Phase {phase}: model returned non-JSON output:\n{llm_output}"
        )

    response_phase = response.get("phase")
    if response_phase != "plan":
        raise RuntimeError(
            f"Step {step}, Phase {phase}: expected phase 'plan', got '{response_phase}'"
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

    return llm_output, plan


def tool_phase(
        messages: list[dict], 
        max_new_tokens: int, 
        completed_steps: list[str], 
        plan: list[str],
        step: int,
        phase: str
        ) -> tuple[bool, list, list]:
 
    if len(completed_steps) > len(plan):
        raise RuntimeError(
            f"Invalid state: completed_steps ({len(completed_steps)}) > plan ({len(plan)})"
        )

    # If all tools are completed - allow final
    if len(completed_steps) == len(plan):
        messages.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "ALL TOOLS COMPLETED.\n"
                    "You MUST now respond with phase='final'.\n"
                    "Tool usage is FORBIDDEN."
                )
            }]
        })
    else:
        # Force next tool
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

    # JSON-only
    try:
        response = json.loads(llm_output)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Step {step}, Phase {phase}: model returned non-JSON output:\n{llm_output}"
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
        
        return True, completed_steps, messages

    else:
        if verbose:
            logger.info(f"Expected phase 'tool', got '{response_phase}'")

        return False, completed_steps, messages


def final_phase(
        messages: list[dict],
        max_new_tokens: int,
        step: int,
        phase: str
        ) -> str:

    messages.append({
        "role": "system",
        "content": [{
            "type": "text",
            "text": (
                "FINAL MODE ACTIVATED.\n"
                "You MUST output ONLY valid JSON.\n"
                "NO markdown. NO explanations.\n"
                "If output is not valid JSON, it will be rejected."
            )
        }]
    })

    llm_output = engine.generate(messages, max_new_tokens)
    
    # JSON-only
    try:
        response = json.loads(llm_output)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Step {step}, Phase {phase}: model returned non-JSON output:\n{llm_output}"
        )

    if response.get("phase") != "final":
        raise RuntimeError(
            f"Expected final answer, got phase '{response.get('phase')}'"
        )

    return response.get("answer", "").strip()


def run_query(
        user_query: str, 
        max_new_tokens_plan=128, 
        max_new_tokens_tool=128,
        max_new_tokens_final=512,
        max_steps=7,
        max_tool_failures=3,
        verbose=False
        ) -> str:
    
    global logger
    logger = setup_logger()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_query}]}
    ]

    plan = None
    completed_steps = []
    phase = "plan"
    
    tool_failures = 0

    for step in range(max_steps):

        # PHASE 1 - PLAN
        if phase == "plan":
            
            start = time.perf_counter()

            llm_output, plan = plan_phase(
                    messages,
                    max_new_tokens_plan,
                    step,
                    phase
            )
            
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": llm_output}]
            })
            
            if verbose:
                elapsed = time.perf_counter() - start
                logger.info(f"run time: {elapsed:.2f} s | Phase 'plan' done.")

            phase = "tool"
            continue

        # PHASE 2 - TOOL EXECUTION
        if phase == "tool":

            start = time.perf_counter()

            tool_response, completed_steps, messages = tool_phase(
                    messages,
                    max_new_tokens_tool,
                    completed_steps,
                    plan,
                    step,
                    phase
            )

            if tool_response:
                tool_failures = 0
                if len(completed_steps) == len(plan):
                    
                    if verbose:
                        elapsed = time.perf_counter() - start
                        logger.info(f"run time: {elapsed:.2f} s | phase 'tool' done.")

                    phase = "final"
                
                continue

            else:
                tool_failures += 1
                
                if verbose:
                    logger.info(f"Tool failure {tool_failures}.")

                if tool_failures >= max_tool_failures:
                    raise RuntimeError(f"Tool phase failed {max_tool_failures} times.")

        # PHASE 3 - FINAL
        if phase == "final":
            
            start = time.perf_counter()
            
            llm_final_output = final_phase(
                    messages,
                    max_new_tokens_final,
                    step,
                    phase
            )

            if verbose:
                elapsed = time.perf_counter() - start
                logger.info(f"run time: {elapsed:.2f} s | phase 'final' done.")

            return llm_final_output
        
    raise RuntimeError("Max steps reached without final answer")

