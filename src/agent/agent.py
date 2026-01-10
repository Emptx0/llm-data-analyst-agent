import json
import time

from .llm import LLMEngine, SYSTEM_PROMPT
from .tools import TOOLS, load_data
from .logger import setup_logger

from src.config import MODEL_ID


engine = LLMEngine(MODEL_ID)


# Returns llm response from the system and raw user prompt.
# Extracts the execution plan from it.
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

    # Prevent duplicate tools (infinite plans)
    if len(plan) != len(set(plan)):
        raise RuntimeError(
            f"Invalid plan: duplicate tools detected: {plan}"
        )

    return llm_output, plan


# If a tool was used in this phase, returns True, the list of completed steps
# and the modified messages.
# Else returns False, the list of completed steps, and the unmodified message.
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
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "All planned tools have been executed.\n"
                    "Now respond with phase='final'.\n"
                    "Do NOT call any tools.\n"
                    "Use ONLY information obtained from tool outputs.\n"
                    "Copy numeric values EXACTLY as returned by tools."
                )
            }]
        })

    else:
        # Force next tool
        expected_tool = plan[len(completed_steps)]

        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    f"Call the tool '{expected_tool}' now.\n"
                    "Respond ONLY with valid JSON matching this schema:\n"
                    "{\n"
                    '  "phase": "tool",\n'
                    f'  "tool": "{expected_tool}",\n'
                    '  "arguments": {}\n'
                    "}\n"
                    "Do NOT answer with phase='final'."
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

    if response_phase != "tool":
        raise RuntimeError(
            f"Step {step}, Phase {phase}: expected phase 'tool', got '{response_phase}'."
        )

    # Tool call
    tool_name = response.get("tool")

    expected_tool = plan[len(completed_steps)]
    if tool_name != expected_tool:
        raise RuntimeError(
            f"Expected tool '{expected_tool}', got '{tool_name}'"
        )

    args = response.get("arguments", {})

    try:
        result = TOOLS[tool_name](**args)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e)
        return False, completed_steps, messages

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


# Returns final llm response
def final_phase(
        messages: list[dict],
        max_new_tokens: int,
        step: int,
        phase: str
        ) -> str:

    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                "All tools are completed.\n"
                "Now respond with the FINAL answer.\n"
                "Use ONLY information obtained from tool outputs.\n"
                "Copy numeric values EXACTLY as returned by tools.\n"
                "Respond ONLY with valid JSON."
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
        dataset_path: str,
        max_new_tokens_plan=128, 
        max_new_tokens_tool=128,
        max_new_tokens_final=512,
        max_steps=7,
        max_tool_failures=3,
        verbose=False
        ) -> str:
    
    global logger
    logger = setup_logger(verbose)
    
    try:
        data_shape = load_data(dataset_path)
    except Exception as e:
        logger.error("Loading data failed: %s", e)
        raise

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_query}]}
    ]

    plan = None
    completed_steps = []
    phase = "plan"
    
    tool_failures = 0

    step_start = None

    for step in range(max_steps):

        step_start = time.perf_counter()
        logger.info(
                "Step %d/%d | Phase '%s' started",
                step+1, max_steps, phase
        )

        # PHASE 1 - PLAN
        if phase == "plan":
            
            llm_output, plan = plan_phase(
                    messages,
                    max_new_tokens_plan,
                    step,
                    phase
            )
            
            logger.info(f"Plan: {plan}")

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": llm_output}]
            })
            
            phase = "tool"

            elapsed = time.perf_counter() - step_start
            logger.info(
                    "Step %d/%d | Phase 'plan' finished in %.2f s",
                    step+1, max_steps, elapsed
            )

            continue

        # PHASE 2 - TOOL EXECUTION
        if phase == "tool":

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
                    phase = "final"
                
                elapsed = time.perf_counter() - step_start
                logger.info(
                        "Step %d/%d | Phase 'tool' finished in %.2f s",
                        step+1, max_steps, elapsed
                )

                continue

            else:
                tool_failures += 1
                
                logger.info(f"Tool failure {tool_failures}.")

                if tool_failures >= max_tool_failures:
                    raise RuntimeError(f"Tool phase failed {max_tool_failures} times.")

        # PHASE 3 - FINAL
        if phase == "final":
            
            llm_final_output = final_phase(
                    messages,
                    max_new_tokens_final,
                    step,
                    phase
            )

            elapsed = time.perf_counter() - step_start
            logger.info(
                    "Step %d/%d | Phase 'final' finished in %.2f s",
                    step+1, max_steps, elapsed
                    )
            return llm_final_output
    

    raise RuntimeError("Max steps reached without final answer")

