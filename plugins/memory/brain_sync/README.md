# Brain Sync Memory Provider

Brain-backed Hermes memory provider.

It reads shared Brain notes, searches Brain for recall, and mirrors Hermes built-in memory writes into the same Brain proposal/final-memory flow used by the Telegram assistant.

## Required env

- `BRAIN_URL`

## Optional env

- `BRAIN_TOKEN` (leave blank only for intentionally unauthenticated local Brain MCP)
- `BRAIN_TOOL_PREFIX` (defaults to `brain`)
- `BRAIN_MEMORY_QUEUE_NOTE`
- `BRAIN_MEMORY_PROPOSAL_MODE`
- `BRAIN_MEMORY_TRUSTED_MODELS`
- `BRAIN_MEMORY_TRUSTED_PROVIDERS`
- `BRAIN_SYNC_TARGET_MEMORY_NOTE`
- `BRAIN_SYNC_TARGET_USER_NOTE`
- `BRAIN_SYNC_STARTUP_NOTES`
