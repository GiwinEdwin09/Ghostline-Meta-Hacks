"""Structured output: Pydantic model -> strict JSON schema, and lenient reply parsing."""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel, ValidationError

# Constraints strict mode may not accept. Pydantic still enforces them when validating a reply.
_DROPPED = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "format",
    "examples",
}


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A self-contained JSON schema for strict structured output.

    Every object lists all its properties as required and forbids extras, `$ref`s are
    inlined, and an optional field becomes `type: [..., "null"]`. Models should not use dict
    fields (strict mode can't describe free-form objects) or tuples.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    return _clean(_inline(schema, defs))


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return _inline({**target, **extra}, defs)
        return {key: _inline(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    return node


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {
        key: (
            {name: _clean(sub) for name, sub in value.items()}  # field names are never dropped
            if key == "properties"
            else _clean(value)
        )
        for key, value in node.items()
        if key not in _DROPPED
    }
    if "const" in node:
        node["enum"] = [node.pop("const")]
    options = node.get("anyOf")
    if options:
        simple = all(isinstance(o, dict) and set(o) <= {"type"} for o in options)
        if simple:
            types: list[str] = []
            for option in options:
                found = option["type"]
                types.extend(found if isinstance(found, list) else [found])
            del node["anyOf"]
            node["type"] = types[0] if len(types) == 1 else types
    if node.get("type") == "object" or "properties" in node:
        properties = node.setdefault("properties", {})
        node["required"] = list(properties)
        node["additionalProperties"] = False
    return node


def text_format_for(model: type[BaseModel]) -> dict[str, Any]:
    """The Responses API `text.format` value for a model (build plan Part 1, check 1)."""
    return {
        "type": "json_schema",
        "name": model.__name__,
        "schema": strict_schema(model),
        "strict": True,
    }


def response_format_for(model: type[BaseModel]) -> dict[str, Any]:
    """The Chat Completions `response_format` value for a model (Part 1, check 6)."""
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "schema": strict_schema(model), "strict": True},
    }


def parse_reply[M: BaseModel](model: type[M], text: str) -> M:
    """Validate a reply as `model`. Also accepts a reply wrapped in a ```json fence."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return model.model_validate_json(text.strip())


def validation_message(error: ValidationError | json.JSONDecodeError) -> str:
    """A compact description of why a reply failed, for the repair call."""
    if isinstance(error, ValidationError):
        lines = [
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in error.errors()
        ]
        return "; ".join(lines[:12])
    return str(error)
