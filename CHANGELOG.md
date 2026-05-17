# Changelog

## Unreleased

- Changing `[pipeline].language` now causes stale notes to be re-ingested automatically on the next `synto ingest --all` or `synto run`. Re-run compile afterward, or use `synto run` to re-ingest and compile in one step.
