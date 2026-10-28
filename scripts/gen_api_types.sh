#!/usr/bin/env bash
# Regenerates web/src/lib/api-types.ts from the FastAPI app's OpenAPI schema.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -c "import json; from ghostline.api.app import create_app; print(json.dumps(create_app().openapi(), indent=2))" > web/openapi.json
cd web && npx openapi-typescript openapi.json -o src/lib/api-types.ts
