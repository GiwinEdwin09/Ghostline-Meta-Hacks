"""Day 0 smoke test (build plan, Part 1).

Calls every external service once, before any feature code depends on it, and prints the
outcome, tokens and cost of each call along with the answers to Part 1's Verify questions.

Run from the repo root (keys come from .env):

    uv run python scripts/smoke.py              # every step that can run
    uv run python scripts/smoke.py --steps 1,6  # only these

Steps 2, 4 and 5 need a phone clip and are added once one exists. Step 8 (upload and
playback on an iPhone) is a manual spike.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# USD per token (build plan, section 0).
WONDER_IN, WONDER_CACHED, WONDER_OUT = 4.00e-6, 0.20e-6, 20.00e-6
SPARK_IN, SPARK_OUT = 1.25e-6, 4.25e-6

# The test image shows this number and shape, so a correct answer proves the model saw it.
IMAGE_NUMBER, IMAGE_SHAPE = 47, "triangle"

READING_SCHEMA = {
    "type": "object",
    "properties": {"number": {"type": "integer"}, "shape": {"type": "string"}},
    "required": ["number", "shape"],
    "additionalProperties": False,
}


@dataclass
class Result:
    step: int
    name: str
    ok: bool = False
    detail: str = ""
    tokens: str = "-"
    cost_usd: float = 0.0
    verify: list[str] = field(default_factory=list)
    seconds: float = 0.0


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines from .env without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def require(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing in .env: {', '.join(missing)}")


def test_image_data_url() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.add_patch(plt.Polygon([[0.1, 0.15], [0.45, 0.15], [0.275, 0.75]], color="tab:red"))
    ax.text(0.72, 0.45, str(IMAGE_NUMBER), fontsize=64, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def meta_client():
    from openai import OpenAI

    require("MODEL_API_KEY")
    base_url = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")
    return OpenAI(base_url=base_url, api_key=os.environ["MODEL_API_KEY"], timeout=300)


def reviewer_client():
    from openai import OpenAI

    require("REVIEWER_BASE_URL", "REVIEWER_MODEL", "REVIEWER_API_KEY")
    base_url = os.environ["REVIEWER_BASE_URL"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return OpenAI(base_url=base_url, api_key=os.environ["REVIEWER_API_KEY"], timeout=300)


def wonder_model() -> str:
    return os.environ.get("PHYSICIST_MODEL", "muse-wonder-1.0")


def wonder_usage(usage) -> tuple[str, float]:
    cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
    reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
    cost = (
        (usage.input_tokens - cached) * WONDER_IN
        + cached * WONDER_CACHED
        + usage.output_tokens * WONDER_OUT
    )
    tokens = f"in {usage.input_tokens} (cached {cached}), out {usage.output_tokens}"
    return f"{tokens} (reasoning {reasoning})", cost


def add_wonder_usage(result: Result, *responses) -> None:
    parts, total = [], 0.0
    for response in responses:
        tokens, cost = wonder_usage(response.usage)
        parts.append(tokens)
        total += cost
    result.tokens = "; ".join(parts)
    result.cost_usd = total


def read_reading(text: str, lenient: bool) -> dict | None:
    """Parse a {"number", "shape"} reply. Lenient parsing also strips a ```json fence."""
    text = text.strip()
    if lenient and text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and set(data) == {"number", "shape"} else None


def step1_wonder_structured(result: Result) -> None:
    client = meta_client()
    schema = {
        "type": "object",
        "properties": {"planet": {"type": "string"}, "g_m_s2": {"type": "number"}},
        "required": ["planet", "g_m_s2"],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model=wonder_model(),
        input="What is the surface gravity on Mars, in m/s^2?",
        text={
            "format": {"type": "json_schema", "name": "gravity", "schema": schema, "strict": True}
        },
        max_output_tokens=4000,
    )
    add_wonder_usage(result, response)
    data = json.loads(response.output_text)
    result.ok = set(data) == {"planet", "g_m_s2"} and 3.5 < float(data["g_m_s2"]) < 4.0
    result.detail = f"reply {data}"
    result.verify.append("Structured output (json_schema) works on Wonder: " + str(result.ok))


def step3_wonder_tool_image(result: Result) -> None:
    client = meta_client()
    image_url = test_image_data_url()
    tool = {
        "type": "function",
        "name": "get_plot",
        "description": "Render a named plot from the current experiment and return it as an image.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    }
    question = {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "Call get_plot with name 'residuals'. Then tell me which number is "
                "written in the plot and which shape is drawn.",
            }
        ],
    }
    first = client.responses.create(
        model=wonder_model(), input=[question], tools=[tool], max_output_tokens=4000
    )
    calls = [item for item in first.output if item.type == "function_call"]
    if not calls:
        add_wonder_usage(result, first)
        result.detail = "Wonder answered without calling the tool"
        return
    call = calls[0]
    history = [question, *(item.model_dump(exclude_none=True) for item in first.output)]
    image_output = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": [{"type": "input_image", "image_url": image_url}],
    }

    from openai import BadRequestError

    try:
        second = client.responses.create(
            model=wonder_model(),
            input=[*history, image_output],
            tools=[tool],
            max_output_tokens=4000,
        )
        route = "image inside function_call_output"
    except BadRequestError as error:
        # Fall back to the orchestrator's plan B: text tool output, image in a user message.
        result.verify.append(f"Image inside a tool output was rejected: {str(error)[:200]}")
        second = client.responses.create(
            model=wonder_model(),
            input=[
                *history,
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "The plot is attached in the next message.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": image_url}],
                },
            ],
            tools=[tool],
            max_output_tokens=4000,
        )
        route = "image in a follow-up user message"
    add_wonder_usage(result, first, second)
    answer = second.output_text
    saw_image = str(IMAGE_NUMBER) in answer and IMAGE_SHAPE in answer.lower()
    result.ok = saw_image
    result.detail = f"called {call.name}({call.arguments}); via {route}; reply: {answer[:160]!r}"
    result.verify.append(
        f"Tool outputs can carry images: {saw_image and route.startswith('image inside')}"
    )
    result.verify.append("Passing the first response's output items back as history works")


def step6_spark_open(result: Result) -> None:
    from openai import BadRequestError

    client = reviewer_client()
    image = {"type": "image_url", "image_url": {"url": test_image_data_url()}}
    question = "Which number is written in this image, and which shape is drawn?"
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "reading", "schema": READING_SCHEMA, "strict": True},
    }

    def ask(text: str, **extra):
        return client.chat.completions.create(
            model=os.environ["REVIEWER_MODEL"],
            messages=[{"role": "user", "content": [{"type": "text", "text": text}, image]}],
            max_tokens=4000,
            extra_headers={"Modal-Routing-Affinity-Key": "smoke-test"},
            **extra,
        )

    # The question doesn't mention JSON, so a JSON reply means the schema was enforced.
    responses, data = [], None
    try:
        responses.append(ask(question, response_format=response_format))
        data = read_reading(responses[-1].choices[0].message.content or "", lenient=False)
        result.verify.append(f"response_format json_schema accepted; enforced: {data is not None}")
    except BadRequestError as error:
        result.verify.append(f"response_format json_schema rejected: {str(error)[:200]}")
    if data is None:
        # The reviewer's fallback: ask for JSON in the prompt and parse the reply.
        responses.append(ask(question + ' Reply with JSON like {"number": 0, "shape": "circle"}.'))
        data = read_reading(responses[-1].choices[0].message.content or "", lenient=True)
    prompt_tokens = sum(r.usage.prompt_tokens for r in responses)
    completion_tokens = sum(r.usage.completion_tokens for r in responses)
    result.tokens = f"in {prompt_tokens}, out {completion_tokens}"
    result.cost_usd = prompt_tokens * SPARK_IN + completion_tokens * SPARK_OUT
    result.ok = data is not None and data["number"] == IMAGE_NUMBER
    result.detail = f"reply {(responses[-1].choices[0].message.content or '')[:160]!r}"
    result.verify.append(f"Reads images: {result.ok}")


SANDBOX_FIT = """
import json, sys
job = json.load(open(sys.argv[1]))
t, y = job["t"], job["y"]
# Least squares for y = g t^2 / 2.
g = 2 * sum(yi * ti**2 for ti, yi in zip(t, y)) / sum(ti**4 for ti in t)
json.dump({"g": g}, open("/work/result.json", "w"))
print("fit done")
"""

NETWORK_PROBE = "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)"


def step7_modal_sandbox(result: Result) -> None:
    import modal

    environment = os.environ.get("MODAL_ENVIRONMENT", "dev")
    app = modal.App.lookup("ghostline-smoke", create_if_missing=True, environment_name=environment)
    sandbox = modal.Sandbox.create(
        app=app,
        image=modal.Image.debian_slim(python_version="3.12"),
        block_network=True,
        timeout=300,
        cpu=1,
        memory=1024,
    )
    try:
        t = [i / 100 for i in range(1, 51)]
        sandbox.filesystem.make_directory("/work")
        sandbox.filesystem.write_text(
            json.dumps({"t": t, "y": [4.905 * x**2 for x in t]}), "/work/job.json"
        )
        sandbox.filesystem.write_text(SANDBOX_FIT, "/work/fit.py")
        fit = sandbox.exec("python", "/work/fit.py", "/work/job.json", timeout=60)
        fit.wait()
        g = json.loads(sandbox.filesystem.read_text("/work/result.json"))["g"]
        probe = sandbox.exec("python", "-c", NETWORK_PROBE, timeout=30)
        probe.wait()
        blocked = probe.returncode != 0
    finally:
        sandbox.terminate()
    result.ok = fit.returncode == 0 and abs(g - 9.81) < 1e-6 and blocked
    result.detail = f"environment {environment}; fit exit {fit.returncode}, g = {g:.4f}"
    result.tokens = "CPU only"
    result.verify.append(f"write_text, exec and read_text round trip works: {fit.returncode == 0}")
    result.verify.append(f"block_network=True stops outbound requests: {blocked}")


STEPS: dict[int, tuple[str, Callable[[Result], None] | None]] = {
    1: ("Wonder, text with structured output", step1_wonder_structured),
    2: ("Wonder with a 10 s video (needs a phone clip)", None),
    3: ("Wonder tool calling with an image output", step3_wonder_tool_image),
    4: ("SAM 3.1 on one image (needs a phone clip)", None),
    5: ("SAM 3.1 on a 3 s video (needs a phone clip)", None),
    6: ("Spark Open on Modal, text plus image", step6_spark_open),
    7: ("Modal Sandbox with the network blocked", step7_modal_sandbox),
    8: ("Upload and playback on iPhone Safari (manual spike)", None),
}


def run(step: int) -> Result:
    name, fn = STEPS[step]
    result = Result(step, name)
    if fn is None:
        result.detail = "not run"
        return result
    started = time.monotonic()
    try:
        fn(result)
    except Exception as error:  # every failure is a finding worth printing
        result.ok = False
        result.detail = f"{type(error).__name__}: {str(error)[:300]}"
    result.seconds = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    runnable = ",".join(str(n) for n, (_, fn) in STEPS.items() if fn)
    parser.add_argument("--steps", default=runnable, help=f"comma-separated (default {runnable})")
    args = parser.parse_args()
    load_env(ROOT / ".env")

    results = []
    for step in (int(s) for s in args.steps.split(",")):
        result = run(step)
        results.append(result)
        status = "PASS" if result.ok else ("SKIP" if result.detail == "not run" else "FAIL")
        print(f"\n[{status}] {step}. {result.name} ({result.seconds:.1f} s)")
        print(f"  {result.detail}")
        print(f"  tokens: {result.tokens}; cost: ${result.cost_usd:.4f}")
        for line in result.verify:
            print(f"  verify: {line}")
    total = sum(r.cost_usd for r in results)
    print(f"\nTotal model cost: ${total:.4f}")


if __name__ == "__main__":
    main()
