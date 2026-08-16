# Repository agent instructions

- Batch independent tool calls whenever possible.
- Do not reread files that have not changed.
- Avoid repeated status polling; report only meaningful state changes.
- Prefer blocking or event-driven waits. If polling is unavoidable, wait at least 60 seconds between polls.
- Delegate builds, tests, CI, deployments, and pull-request monitoring to `operations_watcher`.
- Use the lowest-cost suitable model for routine work. Reserve GPT-5.6 Sol for complex debugging, architecture, and difficult reasoning.
- Never duplicate work owned by `operations_watcher`.

