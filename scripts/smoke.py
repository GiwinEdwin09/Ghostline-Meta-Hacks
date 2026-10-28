"""Day 0 smoke test (build plan, Part 1).

Calls every external service once, before any feature code depends on it, and prints the
outcome, tokens and cost of each call along with the answers to Part 1's Verify questions.

Run from the repo root (keys come from .env):

    uv run python scripts/smoke.py                       # every step that can run
    uv run python scripts/smoke.py --steps 1,6           # only these
    uv run python scripts/smoke.py --clip my_clip.mp4    # steps 2, 4, 5 on a phone clip

Steps 2, 4 and 5 need a clip. Without --clip they render a short synthetic one (OpenCV: a
bouncing ball and a blue dot on a white sheet, 480p) as a stand-in, and the printed answers say
so. A synthetic clip is a proxy: rerun with a phone clip before trusting token counts. Step 8
(upload and playback on an iPhone) is a manual spike.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "tests"))  # fixtures.synthetic_clip

# USD per token (build plan, section 0).
WONDER_IN, WONDER_CACHED, WONDER_OUT = 4.00e-6, 0.20e-6, 20.00e-6
SPARK_IN, SPARK_OUT = 1.25e-6, 4.25e-6
SAM_VIDEO_PER_FRAME, SAM_VIDEO_MIN_FRAMES, SAM_IMAGE = 0.20e-3, 50, 2.50e-3

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


# Steps 2, 4 and 5 work on a clip: a phone clip from --clip, or a synthetic stand-in.


@dataclass
class Media:
    """The clip and the files cut from it."""

    master: Path
    wonder_480: Path  # 10 fps, 480p, at most 10 s: what Part 5 sends to Wonder
    sam_clip: Path  # the first 3 s of the master, at its own frame rate
    frame_png: Path  # one frame from the middle of sam_clip, for the SAM image check
    frame_index: int
    synthetic: bool
    truth: object | None  # ClipTruth, for a synthetic clip
    object_prompt: str
    sheet_prompt: str
    out: Path


MEDIA: Media | None = None


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL)


def clip_seconds(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(probe.stdout.strip())


def prepare_media(clip: Path | None, out: Path, object_prompt: str, sheet_prompt: str) -> Media:
    """Use `clip`, or render the synthetic stand-in, and cut the files steps 2, 4 and 5 send."""
    out.mkdir(parents=True, exist_ok=True)
    truth = None
    if clip is None:
        from fixtures import synthetic_clip

        clip = out / "synthetic_master.mp4"
        truth = synthetic_clip.render_clip(clip, seconds=10, fps=30)
    wonder = out / "wonder_480.mp4"
    ffmpeg("-i", str(clip), "-t", "10", "-vf", "fps=10,scale=-2:480", "-an", "-c:v", "libx264",
           "-crf", "23", "-movflags", "+faststart", str(wonder))  # fmt: skip
    sam_clip = out / "sam_3s.mp4"
    ffmpeg("-i", str(clip), "-t", "3", "-an", "-c:v", "libx264", "-crf", "16", "-fps_mode",
           "passthrough", "-movflags", "+faststart", str(sam_clip))  # fmt: skip
    frame_png = out / "frame.png"
    extract_frame(sam_clip, 45, frame_png)
    return Media(clip, wonder, sam_clip, frame_png, 45, truth is not None, truth,
                 object_prompt, sheet_prompt, out)  # fmt: skip


def extract_frame(video: Path, index: int, path: Path) -> None:
    ffmpeg("-i", str(video), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(path))


def upload_video(client, path: Path) -> str:
    with path.open("rb") as handle:
        uploaded = client.files.create(
            file=(path.name, handle),
            purpose="user_data",
            expires_after={"anchor": "created_at", "seconds": 604800},
        )
    return uploaded.id


def proxy_note() -> str:
    return " (synthetic clip, a proxy)" if MEDIA and MEDIA.synthetic else ""


def step2_wonder_video(result: Result) -> None:
    client = meta_client()
    media = MEDIA
    file_id = upload_video(client, media.wonder_480)
    frames = round(10 * min(10.0, clip_seconds(media.wonder_480)))
    question = "In one short sentence: what moves in this video, and what color is it?"
    replies, notes = [], []
    # The docs allow either part type for an uploaded video, so try both.
    for part_type in ("input_video", "input_file"):
        content = [
            {"type": "input_text", "text": question},
            {"type": part_type, "file_id": file_id},
        ]
        try:
            response = client.responses.create(
                model=wonder_model(),
                input=[{"type": "message", "role": "user", "content": content}],
                reasoning={"effort": "minimal"},
                max_output_tokens=1500,
            )
        except Exception as error:
            notes.append(f"{part_type} rejected ({type(error).__name__}: {str(error)[:140]})")
            continue
        replies.append(response)
        notes.append(f"{part_type}: {response.usage.input_tokens} input tokens")
    try:
        client.files.delete(file_id)
    except Exception:  # the upload expires on its own
        pass
    result.detail = "; ".join(notes)
    if not replies:
        return
    add_wonder_usage(result, *replies)
    first = replies[0]
    result.ok = bool(first.output_text.strip())
    result.detail += f"; reply {first.output_text[:140]!r}"
    tokens = first.usage.input_tokens
    result.verify.append(
        f"Wonder input tokens for a 10 s, 480p, 10 fps video ({frames} frames){proxy_note()}: "
        f"{tokens}, about {tokens / frames:.0f} per frame"
    )
    result.verify.append(
        f"Whole-video input costs about ${tokens * WONDER_IN:.3f} per scene call{proxy_note()}"
    )


def sam_stream(client, media_part: dict, prompt: str, metadata: dict):
    """One streamed SAM call. Returns the raw events (as dicts), the output text and the usage."""
    stream = client.responses.create(
        model=os.environ.get("SAM_MODEL", "sam-3.1"),
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, media_part],
            }
        ],
        stream=True,
        metadata=metadata,
    )
    events, texts, usage = [], [], {}
    for event in stream:
        raw = event.model_dump(mode="json", exclude_none=True)
        events.append(raw)
        if event.type == "response.output_text.delta":
            texts.append(event.delta)
        elif event.type in ("response.completed", "response.incomplete"):
            usage = raw.get("response", {}).get("usage") or {}
    return events, "".join(texts), usage


def decode_raster(record):
    import numpy as np
    from meta_sam_parser import decode_mask_to_raster

    flat = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    return flat.reshape(record.mask.height, record.mask.width)


def paste(raster, left: int, top: int, shape: tuple[int, int]):
    """Place a box-sized raster into a full-frame array with its top-left at (left, top)."""
    import numpy as np

    canvas = np.zeros(shape, np.uint8)
    canvas[top : top + raster.shape[0], left : left + raster.shape[1]] = raster
    return canvas


def centroid(raster, left: float, top: float):
    import numpy as np

    ys, xs = np.nonzero(raster)
    return np.array([xs.mean() + left, ys.mean() + top])


def overlay_png(frame_path: Path, record, out_path: Path) -> None:
    import cv2

    image = cv2.imread(str(frame_path))
    left, top = int(record.bounds.left), int(record.bounds.top)
    mask = paste(decode_raster(record), left, top, image.shape[:2]).astype(bool)
    tinted = image.copy()
    tinted[mask] = (0, 255, 0)
    image = cv2.addWeighted(tinted, 0.55, image, 0.45, 0)
    corner = (int(record.bounds.right) - 1, int(record.bounds.bottom) - 1)
    cv2.rectangle(image, (left, top), corner, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), image)


def box_size(record) -> tuple[int, int]:
    return round(record.bounds.right - record.bounds.left), round(
        record.bounds.bottom - record.bounds.top
    )


def step4_sam_image(result: Result) -> None:
    import cv2
    import numpy as np
    from meta_sam_parser import CompletedOutcome, image_segmentation_format

    client = meta_client()
    media = MEDIA
    data_url = "data:image/png;base64," + base64.b64encode(media.frame_png.read_bytes()).decode()
    _, text, usage = sam_stream(
        client,
        {"type": "input_image", "image_url": data_url},
        media.sheet_prompt,
        {"include_confidence": "true"},
    )
    parser = image_segmentation_format().create_parser()
    parser.push(text)
    parsed = parser.finish(CompletedOutcome()).result
    masks = [r for r in parsed.records if r.kind == "mask"]
    result.cost_usd = SAM_IMAGE
    result.tokens = "1 image"
    result.ok = bool(masks) and not any(d.severity == "error" for d in parsed.diagnostics)
    result.detail = (
        f"{len(masks)} mask records, confidences "
        f"{[round(r.confidence, 3) for r in masks if r.confidence is not None]}; usage {usage}"
    )
    if not masks:
        return
    best = max(masks, key=lambda r: r.mask.width * r.mask.height)
    overlay = media.out / "sam_image_overlay.png"
    overlay_png(media.frame_png, best, overlay)
    left, top = int(best.bounds.left), int(best.bounds.top)
    width, height = box_size(best)
    line = (
        f"mask raster {best.mask.width}x{best.mask.height}, box {width}x{height} at "
        f"({left}, {top}), "
        f"box-sized: {(width, height) == (best.mask.width, best.mask.height)}"
    )
    if media.synthetic:
        truth = np.zeros(decode_shape(media.frame_png), np.uint8)
        cv2.fillConvexPoly(truth, np.round(media.truth.sheet_corners).astype(np.int32), 1)
        raster = decode_raster(best)

        def iou(placed):
            return (truth & placed).sum() / max((truth | placed).sum(), 1)

        line += (
            f"; IoU with the true sheet: {iou(paste(raster, left, top, truth.shape)):.3f} anchored "
            f"at the box's top-left, {iou(paste(raster, 0, 0, truth.shape)):.3f} if anchored "
            "at the frame origin"
        )
    result.verify.append(f"SAM image mask layout{proxy_note()}: {line}; overlay at {overlay}")


def decode_shape(path: Path) -> tuple[int, int]:
    import cv2

    return cv2.imread(str(path)).shape[:2]


def step5_sam_video(result: Result) -> None:
    from meta_sam_parser import (
        CompletedOutcome,
        IncompleteOutcome,
        parse_responses_stream,
        video_segmentation_format,
    )

    client = meta_client()
    media = MEDIA
    file_id = upload_video(client, media.sam_clip)
    try:
        events, text, usage = sam_stream(
            client,
            {"type": "input_video", "file_id": file_id},
            media.object_prompt,
            {"include_confidence": "true"},
        )
    finally:
        try:
            client.files.delete(file_id)
        except Exception:  # the upload expires on its own
            pass
    jsonl = media.out / "sam_video.events.jsonl"
    jsonl.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
    types = sorted({e["type"] for e in events})
    completed = any(e["type"] == "response.completed" for e in events)

    parser = video_segmentation_format().create_parser()
    parser.push(text)
    outcome = CompletedOutcome() if completed else IncompleteOutcome(reason="eof")
    parsed = parser.finish(outcome).result

    async def through_adapter():
        async def source():
            for event in events:
                yield event

        return await parse_responses_stream(source(), video_segmentation_format()).final_result()

    adapter_agrees = len(asyncio.run(through_adapter()).records) == len(parsed.records)

    masks = [r for r in parsed.records if r.kind == "mask"]
    frames = sorted({r.frame.frame_index for r in masks})
    objects = sorted({r.object_id for r in masks})
    billed = usage.get("video_frames_processed")
    result.cost_usd = max(billed or 90, SAM_VIDEO_MIN_FRAMES) * SAM_VIDEO_PER_FRAME
    result.tokens = f"{billed} frames billed"
    result.ok = bool(masks) and not any(d.severity == "error" for d in parsed.diagnostics)
    result.detail = (
        f"{len(events)} events ({', '.join(types)}); {len(masks)} mask records over "
        f"{len(frames)} frames, object ids {objects}; the async adapter agrees with the sync "
        f"parser: {adapter_agrees}; events at {jsonl}"
    )
    result.verify.append(f"SAM video usage: {usage}")
    if parsed.diagnostics:
        result.verify.append(
            f"Parser diagnostics: {[(d.severity, d.code) for d in parsed.diagnostics][:5]}"
        )
    if not masks:
        return

    same_size = sum((r.mask.width, r.mask.height) == box_size(r) for r in masks)
    line = f"{same_size} of {len(masks)} mask rasters have exactly their box's size"
    if media.synthetic:
        import numpy as np

        anchored, at_origin = [], []
        for r in masks:
            truth = media.truth.centers[r.frame.frame_index]
            raster = decode_raster(r)
            anchored.append(np.hypot(*(centroid(raster, r.bounds.left, r.bounds.top) - truth)))
            at_origin.append(np.hypot(*(centroid(raster, 0, 0) - truth)))
        line += (
            f"; median centroid error against the true ball centre: {np.median(anchored):.2f} px "
            f"anchored at the box's top-left, {np.median(at_origin):.1f} px if anchored at the "
            "frame origin"
        )
    sample = masks[len(masks) // 2]
    frame_file = media.out / "sam_video_frame.png"
    extract_frame(media.sam_clip, sample.frame.frame_index, frame_file)
    overlay = media.out / "sam_video_overlay.png"
    overlay_png(frame_file, sample, overlay)
    result.verify.append(f"SAM video mask layout{proxy_note()}: {line}; overlay at {overlay}")
    confidences = [r.confidence for r in masks if r.confidence is not None]
    result.verify.append(
        f"include_confidence on video: {len(confidences)} of {len(masks)} masks carry c"
        + (f", median {statistics.median(confidences):.3f}" if confidences else "")
    )


STEPS: dict[int, tuple[str, Callable[[Result], None] | None]] = {
    1: ("Wonder, text with structured output", step1_wonder_structured),
    2: ("Wonder with a 10 s video", step2_wonder_video),
    3: ("Wonder tool calling with an image output", step3_wonder_tool_image),
    4: ("SAM 3.1 on one image", step4_sam_image),
    5: ("SAM 3.1 on a 3 s video", step5_sam_video),
    6: ("Spark Open on Modal, text plus image", step6_spark_open),
    7: ("Modal Sandbox with the network blocked", step7_modal_sandbox),
    8: ("Upload and playback on iPhone Safari (manual spike)", None),
}
CLIP_STEPS = {2, 4, 5}


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
    parser.add_argument("--clip", type=Path, help="a phone clip for steps 2, 4 and 5")
    parser.add_argument("--object-prompt", default="orange ball", help="SAM prompt for step 5")
    parser.add_argument("--sheet-prompt", default="white sheet of paper", help="prompt for step 4")
    parser.add_argument("--out", type=Path, help="where to write clips, overlays and events")
    args = parser.parse_args()
    load_env(ROOT / ".env")

    steps = [int(s) for s in args.steps.split(",")]
    if CLIP_STEPS & set(steps):
        global MEDIA
        out = args.out or Path(tempfile.gettempdir()) / "ghostline-smoke"
        MEDIA = prepare_media(args.clip, out, args.object_prompt, args.sheet_prompt)
        print(f"clip: {MEDIA.master} ({'synthetic' if MEDIA.synthetic else 'phone'}); out: {out}")

    results = []
    for step in steps:
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
