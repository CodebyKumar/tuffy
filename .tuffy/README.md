# .tuffy/

This directory contains configuration files and custom skills for the Tuffy agent.

## Contents

1. **`skills/`**: Custom capability packs (skills) containing instructions (`SKILL.md`) and optional tools/configs.
2. **`mcp.json`**: Model Context Protocol (MCP) server configurations.
3. **`settings.json`**: Persisted user settings, such as the chosen default model ID.

## Version Control

- `skills/` is tracked in version control so that standard/team-wide capabilities are shared.
- Configurations (`mcp.json` and `settings.json`) are gitignored to prevent machine-specific settings or credentials from being committed.
