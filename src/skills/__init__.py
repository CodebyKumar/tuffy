"""Skills package: discover_skills() (called once at startup by main.py)
scans ./skills/*/ and registers this module's read_skill tool alongside
whatever tools.py each skill ships. See loader.py for the on-disk format."""

from src.tools.registry import registry
from src.skills.loader import discover_skills, list_skills, skill_prompt_lines, mcp_configs_from_skills


@registry.register(
    name="read_skill",
    description="Read the full guidance for a named skill (from the skills list in your prompt) when its "
                "one-line description suggests it's relevant to what the user is asking. Returns detailed "
                "instructions on how to approach that kind of task.",
    parameters={
        "name": {"type": "string", "description": "Exact skill name as shown in the skills list."}
    },
    required=["name"],
    group="docs",
)
def read_skill(name: str) -> str:
    skills = list_skills()
    if name not in skills:
        return f"No such skill '{name}'. Available: {list(skills.keys())}"
    return skills[name]["body"]
