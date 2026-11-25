"""Grip preferences for the simulated gripper."""

FORCE_TARGETS = {"gentle": 2.0, "firm": 5.0}


def grip_mode(instruction: str) -> str:
    # shortcut: use explicit phrases until a language model can be tested against ambiguous prompts.
    instruction = " ".join(instruction.casefold().split()).rstrip(".")
    if instruction.startswith("grip "):
        instruction = instruction[5:]
    instruction = {"gently": "gentle", "firmly": "firm"}.get(instruction, instruction)
    if instruction not in FORCE_TARGETS:
        raise ValueError('Grip instruction must be "grip gently" or "grip firmly".')
    return instruction
