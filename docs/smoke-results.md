# Day 0 smoke test results

Results of `scripts/smoke.py` (build plan, Part 1). Run on 2026-10-28 against the Meta Model API and the `giwin-meta-hacks` Modal workspace, `dev` environment.

## Results

| Step | Check | Result | Tokens | Cost | Time |
|---|---|---|---|---|---|
| 1 | Wonder, text with structured output | Pass | in 22, out 418 (390 reasoning) | $0.0084 | 6.5 s |
| 2 | Wonder with a 10 s video | Not run: needs a phone clip | | | |
| 3 | Wonder tool calling with an image output | Pass | two calls: in 515 / 778 (500 cached), out 134 / 149 | $0.0089 | 25.1 s |
| 4 | SAM 3.1 on one image | Not run: needs a phone clip | | | |
| 5 | SAM 3.1 on a 3 s video | Not run: needs a phone clip | | | |
| 6 | Spark Open on Modal, text plus image | Pass | in 325, out 218 | $0.0013 | 2.9 s |
| 7 | Modal Sandbox with the network blocked | Pass | CPU only | ~$0 | 2.7 s |
| 8 | Upload and playback on iPhone Safari | Not run: manual spike | | | |

Steps 1, 3, 6 and 7 together cost about $0.02.

## Verify answers

- **Wonder structured output (step 1).** `text={"format": {"type": "json_schema", ..., "strict": True}}` on the Responses API works. The prompt didn't mention JSON and the reply still matched the schema.
- **Tool outputs can carry images (step 3): yes.** A `function_call_output` whose `output` is a list holding an `input_image` (a base64 data URL) reached Wonder: it read the number and shape off the image. The orchestrator in Part 13 doesn't need the fallback of attaching images in a follow-up user message.
- **Conversation history (step 3).** Sending the first response's `output` items back as input (`item.model_dump(exclude_none=True)`), followed by the tool output, works. `previous_response_id` isn't needed.
- **Spark Open honors `response_format` with a JSON schema (step 6): yes.** The prompt didn't ask for JSON, and the reply was schema-valid JSON, so the endpoint enforces the schema. The reviewer in Part 14 can rely on it.
- **Spark Open reads images (step 6): yes**, via Chat Completions `image_url` with a base64 data URL.
- **Spark Open connection.** `OpenAI(base_url=f"{REVIEWER_BASE_URL}/v1", api_key=REVIEWER_API_KEY)` with the proxy token in `wk-….ws-…` form works, and the `Modal-Routing-Affinity-Key` header is accepted.
- **Modal Sandbox (step 7).** `Sandbox.create(..., block_network=True)`, `sb.filesystem.write_text`, `sb.exec` and `sb.filesystem.read_text` round-trip correctly. An outbound HTTPS request from inside the sandbox fails, so the network block holds. Note the argument order: `write_text(data, remote_path)`, with the data first.

## Observations

- **Reasoning drives Wonder's cost.** A one-line question used 390 reasoning tokens out of 418 output tokens, billed at the $20 per 1M output rate. The budget guard in Part 23 should count reasoning tokens, and short calls should set a low reasoning effort if Wonder supports it.
- **Prompt caching is automatic.** The second call in step 3 had 500 of its 778 input tokens served from cache, with no cache setting in the request.
- **Latency.** Wonder took 6–13 s per call, and Spark Open about 3 s.

## Still to run

- **Steps 2, 4 and 5** need a phone clip: about 10 s of something moving with an A4 sheet in view. Step 2 measures Wonder's input tokens for a video, which decides between sending the whole video or sampled frames (Part 6). Step 5 checks the SAM mask layout.
- **Step 8** needs an iPhone: chunked upload of a large clip, Range-request playback in Safari, and an `ffprobe` of a 240 fps Slo-Mo upload.
