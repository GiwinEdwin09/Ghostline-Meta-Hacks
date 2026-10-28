# Day 0 smoke test results

Results of `scripts/smoke.py` (build plan, Part 1). Steps 1, 3, 6 and 7 ran on 2026-10-28 against the Meta Model API and the `giwin-meta-hacks` Modal workspace, `dev` environment. Steps 2, 4 and 5 ran on 2026-10-28 on a **synthetic clip** (a bouncing orange ball and a blue dot on a white A4-aspect sheet over a dark background, 854x480, rendered with OpenCV and encoded as H.264 with PyAV), because no phone clip exists yet. A synthetic clip is a proxy: the API-format answers below carry over, but the token and accuracy numbers should be confirmed on a phone clip with `uv run python scripts/smoke.py --clip PATH --object-prompt "..." --steps 2,4,5`.

## Results

| Step | Check | Result | Tokens | Cost | Time |
|---|---|---|---|---|---|
| 1 | Wonder, text with structured output | Pass | in 22, out 418 (390 reasoning) | $0.0084 | 6.5 s |
| 2 | Wonder with a 10 s video (synthetic clip) | Pass | in 1,546 (both part types), out 94 / 78 | $0.0101 | 15.0 s |
| 3 | Wonder tool calling with an image output | Pass | two calls: in 515 / 778 (500 cached), out 134 / 149 | $0.0089 | 25.1 s |
| 4 | SAM 3.1 on one image (synthetic frame) | Pass | 1 image | $0.0025 | 5.0 s |
| 5 | SAM 3.1 on a 3 s video (synthetic clip) | Pass | 90 frames | $0.0180 | 13.1 s |
| 6 | Spark Open on Modal, text plus image | Pass | in 325, out 218 | $0.0013 | 2.9 s |
| 7 | Modal Sandbox with the network blocked | Pass | CPU only | ~$0 | 2.7 s |
| 8 | Upload and playback on iPhone Safari | Not run: manual spike | | | |

Steps 1, 3, 6 and 7 together cost about $0.02, and steps 2, 4 and 5 about $0.03.

## Verify answers

### From the first run (real services, no clip needed)

- **Wonder structured output (step 1).** `text={"format": {"type": "json_schema", ..., "strict": True}}` on the Responses API works. The prompt didn't mention JSON and the reply still matched the schema.
- **Tool outputs can carry images (step 3): yes.** A `function_call_output` whose `output` is a list holding an `input_image` (a base64 data URL) reached Wonder: it read the number and shape off the image. The orchestrator in Part 13 doesn't need the fallback of attaching images in a follow-up user message. This was re-checked through `ghostline.llm` when the replay fixtures were recorded.
- **Conversation history (step 3).** Sending the first response's `output` items back as input (`item.model_dump(exclude_none=True)`), followed by the tool output, works. `previous_response_id` isn't needed. The same holds for a structured-output repair call, which sends Wonder's own reasoning and message items back (checked live).
- **Spark Open honors `response_format` with a JSON schema (step 6): yes.** The prompt didn't ask for JSON, and the reply was schema-valid JSON, so the endpoint enforces the schema. The reviewer in Part 14 can rely on it.
- **Spark Open reads images (step 6): yes**, via Chat Completions `image_url` with a base64 data URL.
- **Spark Open connection.** `OpenAI(base_url=f"{REVIEWER_BASE_URL}/v1", api_key=REVIEWER_API_KEY)` with the proxy token in `wk-….ws-…` form works, and the `Modal-Routing-Affinity-Key` header is accepted.
- **Modal Sandbox (step 7).** `Sandbox.create(..., block_network=True)`, `sb.filesystem.write_text`, `sb.exec` and `sb.filesystem.read_text` round-trip correctly. An outbound HTTPS request from inside the sandbox fails, so the network block holds. Note the argument order: `write_text(data, remote_path)`, with the data first.

### From the synthetic clip (steps 2, 4, 5)

- **Wonder's input tokens for video (step 2): about 1,550 for 10 s at 480p and 10 fps (100 frames), 15 tokens per frame of the file.** That is about $0.006 of input, so the whole video can be sent, as Part 6 plans. Both `{"type": "input_video", "file_id": ...}` and `{"type": "input_file", "file_id": ...}` are accepted for a file uploaded with `purpose="user_data"` and `expires_after`, and both gave the same count. This came from a synthetic clip. Token counts depend on duration and resolution more than content, but confirm on a phone clip.
- **Wonder samples video at about one frame per second.** A probe asked Wonder how many distinct frames it could see in the 10 s clip: it listed 10, at roughly 0 s, 1 s, 2 s and so on, which is about 150 tokens per sampled frame. Anything faster than that (a bounce, a fast oscillation) falls between frames: on the synthetic bouncing-ball clip the first scene analysis said "no bounce is visible". Because of this, `analyze_scene` also sends three exact stills with their times and tells Wonder it sees about 1 frame per second; with that note it called the system `bounce` and said a rapid bounce can't be ruled out from the preview. Stills are dearer than video frames: 20 stills at 480p cost 10,259 input tokens (about 510 each, $0.04). For fast systems, add stills across the motion window, up to 50 per request.
- **SAM mask raster layout (steps 4 and 5): the raster is exactly the box's size and is anchored at the box's top-left corner.** This settles the Part 31 risk. On the video, 90 of 90 mask rasters matched their box's width and height. Pasting each raster at its box's top-left gave a median centroid error of 0.11 px against the true ball centre (maximum 0.27 px), against about 405 px if the raster were anchored at the frame origin. On the image, the sheet mask (564x394, box at (141, 46)) had an IoU of 0.986 with the true sheet polygon (0.488 if anchored at the origin). The overlays were checked by eye too. The mask token always carries `x=0;y=0`, which `meta-sam-parser` requires, so there is no other offset to apply.
- **Centroid noise (synthetic).** A static blue dot jittered 0.044 px RMS over 90 frames, and the moving ball was tracked on all 90 frames with sub-pixel centroids. This is on a clean synthetic render, so it says the pipeline adds no offset or noise, not that a phone clip will do as well.
- **SAM usage fields.** A video response reports `usage.video_frames_processed` (90 here), an image response `usage.images_processed` (1), and the token fields are zero. The cost meter in Part 23 bills `max(frames, 50)` per video request from that field.
- **`include_confidence`.** With `"true"`, every box and mask token carries `c`. On the synthetic clip it was the same value on every frame for one object (0.808 for the ball, 0.887 for the dot, 0.922 for the sheet), so treat `c` as an object-level score rather than a per-frame one.
- **Event stream shape (step 5).** One response is 98 events for 90 frames: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, one `response.output_text.delta` per frame (one line each), then `response.output_text.done`, `response.content_part.done`, `response.output_item.done` and `response.completed`, each of which repeats the whole output text. A stream is therefore about 280 KB for 90 frames, four fifths of it duplicates. The parser needs the finalizers kept (it checks that they match), so Ghostline stores the stream unchanged. The sync parser and the async `parse_responses_stream` adapter produced the same records.
- **Object IDs.** The video response used object ID `0` for the single ball, and frames came in ascending order, as documented.

## Observations

- **Reasoning drives Wonder's cost.** A one-line question used 390 reasoning tokens out of 418 output tokens, billed at the $20 per 1M output rate. The budget guard in Part 23 counts reasoning tokens as output. A scene analysis of the synthetic clip (3,394 input and 2,010 output tokens, 1,233 of them reasoning, at medium effort) cost $0.054, and the two rescue calls for the SAM retry ladder (low effort) $0.010 and $0.013. A Spark Open review call cost about $0.002.
- **Prompt caching is automatic.** The second call in step 3 had 500 of its 778 input tokens served from cache, with no cache setting in the request.
- **Latency.** Wonder took 6–13 s per call for short answers and about 15 s for a scene analysis, SAM about 13 s for 90 frames, and Spark Open about 3 s.
- **Wonder is cheap for video, dear for reasoning.** For a scene call, input is about $0.01 and reasoning output about $0.04, so tokens spent thinking, not watching, set the price.

## Still to run

- **A phone clip.** Rerun steps 2, 4 and 5 with `--clip` to confirm the token count and mask layout on real footage, and to check the Part 7 "done when" items that need one (a 600-frame drop tracked continuously, a mask overlay lining up in the browser).
- **Step 8** needs an iPhone: chunked upload of a large clip, Range-request playback in Safari, and an `ffprobe` of a 240 fps Slo-Mo upload.
