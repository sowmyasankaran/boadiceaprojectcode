
import os

AWAITING_INPUT_SENTINEL = "\x00AWAITING_INPUT\x00"


def ask_user(prompt: str = "You: ") -> str:
    """Drop-in replacement for input(prompt) everywhere in the pipeline.
    Every existing `input("You: ")` / `input(f"...: ")` call site should use
    this instead -- see agent.py, intake.py, audit.py for the call sites
    this replaced.

    AGENT_WEB_MODE unset (default, plain `python3 intake.py` in a terminal):
    identical to input(prompt) - same prompt text, same blocking read, no
    sentinel emitted.

    AGENT_WEB_MODE=1 (set by web_backend.py when it spawns the subprocess):
    flushes one line -- the sentinel followed by the prompt text -- then
    reads a line from stdin with no prompt of its own (the prompt was
    already emitted on the sentinel line, so a second one would duplicate
    it). The backend strips the sentinel prefix if it wants the prompt text;
    in practice it doesn't need to, since every input() site in this
    pipeline is already preceded by an explicit `print(f"Agent: ...")` --
    the "You: " style prompts are a CLI-only affordance, not new agent
    content.
    """
    if os.environ.get("AGENT_WEB_MODE"):
        print(f"{AWAITING_INPUT_SENTINEL}{prompt}", flush=True)
        return input()
    return input(prompt)