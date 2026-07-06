# src/skills/

The skills *mechanism* — discovery, loading, and the `read_skill` tool. For the skills
*content* (droppable capability packs), see [/skills](../../skills/) at the repo root.

- [loader.py](loader.py) - `discover_skills()` scans `./skills/*/`, parses each `SKILL.md`'s
  YAML frontmatter + body, auto-imports each skill's optional `tools.py` as a side effect, and
  returns `{name: {description, body, path}}`. Also exposes `skill_prompt_lines()` (one line
  per skill for the system prompt) and `mcp_configs_from_skills()` (each skill's optional
  `mcp.json`, merged into the MCP client's server list).
- [__init__.py](__init__.py) - registers the `read_skill` tool (group `docs`), which fetches a
  named skill's full guidance body on demand.

## Why the full body isn't always in the prompt

Only each skill's one-line description goes into the system prompt via `skill_prompt_lines()`.
The full `SKILL.md` body is fetched only when the model calls `read_skill(name)` and decides
it's relevant — this keeps the prompt small as skills accumulate, the same pattern used for
Claude Code's own skills.

## Loading order at startup

`main.py` calls `discover_skills()` before the first system prompt is built (so skill
descriptions and skill-provided tools are both present from turn one), then
`connect_mcp_servers(extra_configs=mcp_configs_from_skills())` (so a skill's own `mcp.json`
server is known before MCP tools are registered).
