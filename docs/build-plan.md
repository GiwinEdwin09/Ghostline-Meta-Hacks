# Ghostline build plan

Written Tuesday, Oct 27, 2026. Submissions close Wednesday, Nov 4, 9 PM ET.

Each part below has the same shape:

- **What it does:** the job of this piece, in plain terms.
- **How to build it:** the concrete design, libraries, algorithms and edge cases.
- **Done when:** a check you can run to know it works.

Facts marked **(docs)** were checked against Meta Model API docs (dev.meta.ai) and Modal docs (modal.com/docs) on Oct 27. Items marked **Verify** are assumptions to confirm in the Day 0 smoke test (Part 1) before building on them.

---

## 0. The whole system on one page

### What the user does

1. Opens Ghostline on their phone and taps **New experiment**.
2. Picks what they're filming (pendulum, ramp, drop, bounce, spin-down, or "not sure") and follows a five-item setup checklist.
3. Records the motion with the phone's camera app, with an A4 sheet in the frame, then uploads the clip.
4. Watches the pipeline run: Normalizing → Watching → Tracking → Calibrating → Hypothesizing → Fitting → Peer review. This takes about 1–3 minutes.
5. Reads the result:
   - the winning law and its numbers with uncertainty,
   - the hypotheses that were rejected and why,
   - the reviewer's verdict,
   - the lab notebook.
6. Taps **Prove it**, films a second trial, and gets a video with the ghost overlaid, plus an error readout.
7. Opens **What if** and drags sliders: Mars gravity, a steeper ramp, a heavier bob.
8. Taps **Export to simulator** to download the measured friction, bounciness and damping as a file a robotics simulator can load.

### Architecture

```
Phone or laptop browser
  Next.js app on Vercel
      |  chunked upload (8 MiB chunks)          |  poll state every 1 s
      v                                          v
FastAPI on Modal (asgi_app, CPU) ---- spawn ---> Pipeline workers (Modal Functions, CPU: ffmpeg, PyAV, OpenCV)
      |                                              |-- Meta Model API (api.meta.ai/v1)
      |                                              |      Files API, sam-3.1, muse-wonder-1.0
      |                                              |-- Modal Sandbox (fitting harness, network blocked)
      |                                              '-- Modal Shared Endpoint: Muse Spark Open (reviewer)
      v                                              v
Supabase Postgres (experiments, events, fits, costs)     Modal Volume "ghostline-data" (videos, plots, tracks)
```

### Design decisions everything else follows

1. **The phone is only the camera and the screen.** All computation runs on Modal and the Meta Model API.
2. **Wonder writes physics; a trusted harness keeps score.**
   - Muse Wonder proposes models and can write model code.
   - Fitting, held-out scoring and every number shown to the user come from deterministic Python code that Wonder can't change.
   - Wonder never sees the held-out data.
3. **Models come from a library first.**
   - Every system has a curated library of competing models.
   - Wonder can add custom models through a strict interface. Library-only mode is the fallback when anything breaks.
4. **Fit positions, not derivatives.** Differentiating noisy video positions amplifies noise. Velocities are only for display.
5. **Held-out data is the future.**
   - Fit on the first 65% of the motion and score the prediction of the last 35%.
   - A model only wins if its advantage is larger than its uncertainty.
6. **Pair Wonder with SAM 3.1, the way Meta recommends.**
   - Meta's own docs suggest letting a Muse model interpret the scene and decide what to segment, then handing the concept to SAM 3.1 (docs).
   - SAM 3.1 takes one concept per request (docs).
   - The A4 sheet doesn't move, so it's segmented from a single still frame, which is cheap and precise.
7. **Poll instead of streaming to the browser.**
   - Poll every 1 s.
   - This works when mobile Safari puts the tab in the background, and it avoids the 150 s request limit on Modal web endpoints (docs).
8. **Record every model call and allow replay.** Reruns, tests and the demo fallback cost nothing and give the same output every time.
9. **Ship seeded example experiments.** Judges can explore finished results instantly, even if an upstream API is down.
10. **Report honest uncertainty.**
    - Every reported number carries statistical, calibration and timing uncertainty.
    - Ghostline says "can't tell" when the data can't distinguish models.

### Cost per experiment (estimate)

| Item | Basis | Typical cost |
|---|---|---|
| SAM 3.1 video tracking | $0.20 per 1,000 frames, minimum 50 frames per request (docs). A 10 s clip at 60 fps is 600 frames per object. | $0.12 per object at 60 fps; $0.48 at 240 fps |
| SAM 3.1 sheet segmentation | $2.50 per 1,000 images (docs) | under $0.01 |
| Muse Wonder (scene, agent loop, diagnosis) | $4.00 per 1M input, $0.20 per 1M cached input, $20.00 per 1M output (docs) | **~$0.50–1.50, an estimate. Measure it in Part 1.** |
| Muse Spark Open reviewer | $1.25 per 1M input, $4.25 per 1M output on the Modal Shared Endpoint | ~$0.02–0.05 |
| Modal CPU and Sandbox | per-second billing | ~$0.02 |

**Meta Model API spend is separate from your Modal credits.** Wonder is the expensive part. Plan for roughly $80–120 of Model API usage across development, benchmark runs and judging, and find out whether the hackathon provides Model API credits. Part 23 covers the budget guard and replay cache that keep this under control.

---

# A. Foundations

## Part 1: Accounts, keys and the Day 0 smoke test

**What it does:** gets every external service working before any feature code exists, and answers the open questions that change the design.

**How to build it:**

1. **Meta Model API**
   - Create a key and export it as `MODEL_API_KEY`.
   - Rate limits are per team: `muse-wonder-1.0` allows 3,000 RPM and 4M TPM, and `sam-3.1` allows 3,000 RPM (docs).
   - Files uploaded through the Files API never expire unless you set `expires_after`, and a team can store 100 GiB in total (docs). Always upload with a 7-day expiry.
2. **Modal**
   - `pip install modal`, then `modal setup`.
   - Create two environments, `dev` and `main`, so judging traffic never shares state with development.
   - Create a proxy token with `modal workspace proxy-tokens create`. Join the ID and secret with a period: `wk-<id>.ws-<secret>` is the bearer token (docs).
   - Create a Shared Endpoint for Muse Spark Open from the dashboard's Endpoints tab.
   - Record the endpoint URL and the model name shown in the dashboard's request examples. You can also list it with `GET <endpoint>/v1/models` (docs).
3. **Supabase:** create a project (free tier). Copy the pooled connection string (transaction mode, port 6543) as `DATABASE_URL`.
4. **Vercel and GitHub:** create the repo `ghostline` and a Vercel project pointing at `web/`.
5. **Local tooling:**
   - Python 3.12 with `uv`.
   - Node 22 LTS. The `@meta-sam/*` packages need Node ^20.17 or ≥22.9 and are ESM only (docs).
   - `brew install ffmpeg`.
6. **Write `scripts/smoke.py`.** It makes eight calls, prints the outcome, tokens and cost of each, and settles every **Verify** item:
   1. Wonder, text only. Use structured output with a tiny JSON schema.
   2. Wonder with a 10 s, 480p MP4 uploaded through the Files API (`purpose="user_data"`, referenced as `input_file` or `input_video`) (docs). **Record the input token count. It sets whether Wonder gets the whole video or sampled frames (Part 6).**
   3. Wonder with tool calling. Return an image from a tool. **Verify** whether tool outputs can carry images. If they can't, the orchestrator attaches images in a follow-up user message.
   4. SAM 3.1 on one image, using the concept prompt "white sheet of paper".
   5. SAM 3.1 on a 3 s video, with `include_confidence: "true"` and streaming on. Parse the output with `meta-sam-parser`, decode one mask and overlay it on the frame. **Verify** that the mask raster is box-sized and anchored at the box's top-left corner. The docs' examples show the mask raster matching the box's size.
   6. Spark Open on Modal, with text plus one image. **Verify** whether the endpoint honors `response_format` with a JSON schema.
   7. Create a Modal Sandbox with `block_network=True`, write a file with `sb.filesystem.write_text`, run `sb.exec("python", ...)`, and read the result back (docs).
   8. **Upload and playback spike:**
      - Upload a 150 MB clip from an iPhone in Safari to a throwaway Modal endpoint using 8 MiB chunks.
      - Play a video back on iPhone Safari from a FastAPI route that supports Range requests.
      - Upload a 240 fps slow-mo clip from Photos through Safari's file picker, then run `ffprobe` to see what iOS actually delivered (Part 5).

**Done when:** all eight calls succeed, and the token, cost and **Verify** answers are written into `docs/smoke-results.md`.

## Part 2: Repository layout and tooling

**What it does:** one repo, with clear module boundaries, so each part below maps to one folder.

**How to build it:**

```
ghostline/
  README.md
  web/                          Next.js app (Part 24)
    src/app/                    routes: /, /examples, /new, /e/[id], /benchmark, /about, /admin
    src/components/             VideoPlayer, GhostPlayer, ModelCard, NotebookTimeline, WhatIf, ...
    src/lib/api.ts              typed client (generated types from FastAPI OpenAPI)
    src/lib/api-types.ts        generated by openapi-typescript, never hand-edited
    src/lib/sim/                TypeScript twins of the physics library + RK4 integrator (Part 18)
    src/lib/sam/                replay stored SAM event streams into @meta-sam/parser
  backend/
    modal_app.py                Modal App: images, Volume, secrets, functions, asgi_app
    migrations/                 SQL migrations for Postgres
    ghostline/
      api/                      FastAPI routers, auth, Range-enabled media route
      jobs/                     ingest_trial, analyze_experiment, prove_trial, revise, simulate
      ingest/                   ffprobe, timing model, normalization, auto-trim (Part 5)
      scene/                    Wonder scene analysis (Part 6)
      vision/                   SAM client, parser glue, mask features, track stitching (Part 7)
      calib/                    sheet corners, homography, offset correction, gravity (Part 8)
      traj/                     world transform, cleaning, events, noise, out-of-plane check (Part 9)
      physics/
        library/                one module per system: freefall, pendulum, ramp, bounce, spin (Part 10)
        harness/                code that runs inside the Sandbox (Part 11)
        select.py               held-out selection and uncertainty (Part 12)
      agent/                    Wonder orchestrator, tools, budgets (Part 13)
      review/                   Spark Open reviewer (Part 14)
      confidence.py             confidence rules (Part 15)
      ghost/                    prediction, sprite, render, divergence (Parts 16-17)
      export/                   simulator export formats and the PyBullet replay check (Part 19)
      notebook/                 event writer (Part 20)
      llm/                      clients, retries, record/replay cache, cost meter (Part 23)
      store/                    Postgres repo, Volume paths
      prompts/                  versioned prompt files (Appendix B)
    tests/
  synth/                        synthetic video generator for tests (Part 26)
  bench/                        manifest.csv, ground truth, runner, results (Part 25)
  evals/                        prompt evals on real clips (Part 26)
  scripts/                      smoke.py, seed_examples.py, backup.sh
  docs/                         smoke-results.md, architecture.png, prompts changelog
```

- **Python:** `ruff` for lint and formatting, `pytest`, `pydantic` v2 for every schema.
- **TypeScript:** `eslint`, `prettier`, `vitest`.
- **CI:** GitHub Actions runs Python tests in replay mode (no network) plus `vitest`, on every push.
- **One source of truth for the API shape:** FastAPI's OpenAPI spec, turned into `web/src/lib/api-types.ts` (regenerate with `npm run gen:api`) with `openapi-typescript`.

**Done when:** `uv run pytest` and `npm test` pass in CI on an empty skeleton.

## Part 3: Filming kit and capture rules

**What it does:** makes the footage measurable. Most accuracy problems are created at the camera, not in the code.

**How to build it:** this is a physical checklist. The same rules appear in the app's setup screen (Part 4).

**Kit**
- Phone tripod, or a stack of books.
- A4 sheets, plus masking tape.
- A dark board or cloth behind the sheet, for contrast.
- Tape measure.
- A second phone, for stopwatch and audio.
- Objects:
  - a dense small ball (golf ball or steel ball),
  - a foam ball,
  - a bouncy ball,
  - string and a heavy nut, for the pendulum,
  - a 60–100 cm board, books, and a wooden block, for the ramp,
  - a solid ball, for rolling,
  - a fidget spinner with a colored dot sticker on one arm.

**Rules for every clip**
1. **Camera:**
   - Use a tripod; never film handheld.
   - Use the 1x lens; the 0.5x ultra-wide distorts.
   - Film in landscape.
   - Tap and hold to lock focus and exposure.
   - Turn on Grid and Level (iOS: Settings > Camera) and level the phone, so the image's vertical matches true vertical.
2. **Distance:**
   - Stand at least 2 m back, and use the 2x lens if the phone has one. Distance reduces the perspective error from objects moving in front of the sheet (Part 8).
   - The motion should fill at least half the frame width.
3. **Sheet:**
   - Flat, taped at the corners, fully visible the whole time, against a contrasting background.
   - In the plane of motion, or as close to it as possible. Write down how far in front of the sheet the object moves.
4. **Frame rate:**
   - The default is 60 fps normal video (iOS: Settings > Camera > Record Video > 1080p at 60 fps).
   - Use 240 fps slow-mo only for free fall, bounces and fast spinners. See Part 5 for the slow-mo export problem.
5. **Lighting:**
   - Use daylight or bright, steady light.
   - Under LED or fluorescent light, slow-mo shows flicker bands.
6. **Clip:**
   - 3–15 s, longer for spin-down.
   - Start recording before the release and stop after the motion ends.

**Per system**

| System | Setup | Notes |
|---|---|---|
| Free fall | Drop the dense ball and the foam ball 1–1.5 m beside a wall with the sheet on it. The ball should fall 2–3 cm in front of the wall. | The foam ball is what makes the drag models win. |
| Pendulum | String at least 50 cm, a heavy nut, a rigid pivot. One clip at under 15° amplitude, one at about 40°. Tape-measure L from the pivot to the bob's center. | The large-amplitude clip is where the nonlinear model wins. |
| Ramp | Board at 10–30°, filmed side-on, with the sheet on a board behind it parallel to the ramp's vertical plane. Slide the block; roll the solid ball. | The ball should give a = (5/7) g sin α. |
| Bounce | Bouncy ball dropped from about 1 m next to the wall, at 240 fps. Record normal-speed audio on a second phone. | Audio timing is the ground truth (Part 25). |
| Spin-down | Spinner flat on a table, camera directly above, sheet flat on the table. Dot sticker on one arm. Gentle flick, up to about 10 rev/s, at 60 fps. | Angles need no scale, only timing. Keep the clip under 15,000 frames (250 s at 60 fps), SAM's per-request limit (docs). |

**Done when:** three test clips (drop, pendulum, spinner) are filmed on Day 0 following these rules.

---

# B. Measurement pipeline

## Part 4: Capture and upload in the web app

**What it does:** gets the clip from the phone to the back end reliably, with enough context (system, capture mode, plane offset) to measure it correctly.

**How to build it:**

- **Route `/new`, a four-step wizard:**
  1. **What are you filming?** Six cards plus "Not sure, let Ghostline decide".
  2. **Set up.** A checklist for the chosen system, with a small diagram and a one-line "why it matters" per item. It also asks:
     - "How far in front of the sheet does the object move?" in cm, defaulting to 2 cm. This feeds Part 8.
     - Paper size, A4 or US Letter. It's auto-detected in Part 8, so this is only a confirmation.
     - Optionally, "Measured pendulum length (cm)".
  3. **Upload.**
     - Use `<input type="file" accept="video/*">` with no `capture` attribute, so users can choose an existing clip from Photos.
     - Show the file size, an upload-time estimate and a "use Wi-Fi" hint.
     - Reject files over 1 GiB, which is the Meta Files API limit (docs).
  4. **Confirm.** Once ingest finishes (Part 5), show:
     - the playback proxy,
     - an auto-suggested trim range on a two-handle slider,
     - the capture mode and fps that Part 5 detected, with a warning if the clip looks re-timed.
     - The user taps **Analyze**.
- **Chunked, resumable upload.** This avoids the 150 s Modal request limit on slow mobile connections and survives network drops.
  1. Call `POST /api/uploads` to get an `upload_id`.
  2. `PUT /api/uploads/{id}/chunks/{n}` with 8 MiB `Blob.slice` chunks, three in parallel. Retry each chunk up to five times with backoff.
  3. Call `POST /api/uploads/{id}/complete`.
  4. Use `XMLHttpRequest` so you get upload progress events.
  5. Keep the `upload_id` and the received chunk numbers in `sessionStorage`, inside try/catch, so a page reload resumes the upload.
- **Stretch: record in the app.**
  - Use `getUserMedia` with `{ width: 1920, height: 1080, frameRate: 60 }` and `MediaRecorder`.
  - It sidesteps the Photos transcoding problem, but exposure control is worse.
  - Chrome's WebM output must be transcoded server-side, because SAM 3.1 rejects VP9 and WebM (docs).

**Done when:** a 300 MB clip uploads from an iPhone on cellular, survives toggling airplane mode mid-upload, and resumes.

## Part 5: Ingest, normalization and the timing model

**What it does:**
- Turns whatever the phone produced into one clean "analysis master" video with:
  - the correct orientation and SDR color,
  - identical frame count to the source,
  - a trustworthy real-time timestamp for every frame.
- Also produces cheaper copies for playback and for Wonder.

**How to build it:**

Modal Function `ingest_trial`, with `cpu=4`, `memory=8192`, `timeout=900`.

1. **Assemble the chunks** into `original.<ext>` and compute its sha256, which is the cache key for everything downstream.
2. **Probe the file:**
   - Run `ffprobe -show_streams -show_format -show_frames -of json` (for long clips, use `-show_entries frame=best_effort_timestamp_time`).
   - Collect codec, dimensions, rotation (display matrix side data), color transfer, `r_frame_rate`, `avg_frame_rate`, `nb_frames`, duration, all tags, and every frame's timestamp.
   - HDR detection: `color_transfer` of `arib-std-b67` is HLG, which iPhones use by default; `smpte2084` is PQ.
   - Android slow-mo files may carry a capture-rate tag such as `com.android.capture.fps`. Read all tags, and let known capture-rate tags override the container rate.
3. **Build the timing model.** This is the most important step in the file.
   - Compute the frame intervals Δt_i from the timestamps. Classify them as:
     - **CFR:** the standard deviation of Δt is under 2% of its mean.
     - **VFR jitter:** normal for phones. Keep the per-frame timestamps; never assume a constant rate.
     - **Mixed rate:** the intervals form two or more clusters. iOS slow-mo exported with normal-speed ramps at each end looks like this.
   - Combine three sources: what the user declared (normal or slow-mo, and the fps), the metadata, and the interval analysis. That gives `time_factor`, the real seconds per file second:
     - Declared normal, file fps ≈ declared fps: `time_factor = 1`.
     - Declared slow-mo at N fps, intervals ≈ 1/N: this is an original file, so `time_factor = 1`.
     - Declared slow-mo at N fps, intervals uniform at about 1/30 (or 1/25 or 1/60): the clip was re-timed, so `time_factor = playback_fps / N`. Show a warning the user must confirm.
     - Mixed rate: restrict the usable range to the largest uniform cluster (the slowed section) and apply that cluster's factor. Tell the user, and suggest exporting the clip with the whole thing slowed, or uploading the original from a computer.
   - **Physics sanity check:** if any free-flight segment fits g ≈ 9.81/k² for k in {2, 4, 8}, flag "this clip looks re-timed by k×".
4. **Normalize to `master.mp4`, the only file every later step reads:**
   - Bake in rotation. Re-encoding with ffmpeg applies the display matrix; make sure the output carries no rotation tag.
   - Tone-map HDR to SDR BT.709:
     ```
     zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p
     ```
   - Cap the long side at 1920 px.
   - Encode with `libx264 -crf 18 -preset veryfast`.
   - Use `-fps_mode passthrough` so no frame is duplicated or dropped.
   - Use `-movflags +faststart`; SAM's uploader remuxes otherwise (docs).
   - Use `-an` (no audio).
   - **Assert that the master's frame count equals the source's.** SAM frame index N must equal our frame index N, which must equal timestamp t_N.
5. **Derived files:**
   - `playback_720.mp4`: H.264 with faststart, for the browser.
   - `wonder_480.mp4`: 480p, down-sampled to 10 fps, for Wonder to watch. Exact timing doesn't matter here; the prompt tells Wonder the real duration.
   - `audio.wav`: mono, 48 kHz, if there is an audio track. Used for bounce timing.
   - `poster.jpg`.
6. **Auto-trim suggestion:**
   - Decode at 160 px grayscale and compute the absolute difference between consecutive frames.
   - The motion window is where the difference energy exceeds median + 5×MAD, padded by 0.3 s on each side.
7. **Decode with PyAV, not OpenCV.** PyAV returns each frame's presentation timestamp, so frame index and time come from the same pass.
   - Don't write every frame to disk: 2,400 frames at 1080p is close to 1 GB.
   - Decode on demand, and cache a few keyframes as JPEG.
8. **Upload to Meta Files API**, with `purpose="user_data"` and `expires_after={"anchor":"created_at","seconds":604800}` (docs):
   - the trimmed `master.mp4`, for SAM,
   - `wonder_480.mp4`, for Wonder.
   - Store both file IDs.
9. **Write outputs:**
   - `probe.json`
   - `timestamps.npy` (real seconds, after `time_factor`)
   - `timing.json` (classification, factor, source, jitter, warnings)
   - the trial row in Postgres.

**Watch out for:**
- **Edge cases:**
  - Portrait clips are fine after rotation.
  - Very dark clips: warn the user.
  - Clips longer than 15,000 frames after trimming: ask the user to trim (SAM limit, docs).
  - VP9 or WebM: transcode.
  - A missing audio track is fine.
- **What iOS actually delivers through Safari's picker is the biggest unknown** (Part 1, check 8). Defaulting to 60 fps normal video avoids most of it.

**Done when:** each of these produces a correct master with an identical frame count, correct orientation, SDR color and correct real timestamps:
- an iPhone HEVC HDR portrait clip,
- an Android clip,
- a re-timed iPhone slow-mo.

## Part 6: Scene analysis with Muse Wonder

**What it does:**
- Wonder watches the clip and decides what to measure:
  - the moving object, any markers or pivot, the calibration sheet, the kind of system, and quality problems.
- It writes SAM 3.1 prompts.
- Before seeing any data, it states its initial hypotheses. That's what makes the refutation moment auditable later.

**How to build it:**

- **Call:** Responses API, `model="muse-wonder-1.0"`, structured output with the `SceneAnalysis` schema (Appendix C), and streaming on.
  - Wonder is in preview. Pin the model ID, and handle `model_not_found` (docs) with a clear error.
- **Input:**
  - the system prompt from Appendix B,
  - the user's hint and description,
  - the real duration and the declared capture mode,
  - `wonder_480.mp4` as `input_video` with `file_id`,
  - three full-resolution stills (the start, middle and end of the trim) as `input_image` data URLs, for detail.
  - If the Day 0 token count for video is high, send 8–12 sampled frames instead. A request can carry up to 50 images (docs).
- **SAM prompt rules,** written into the prompt from Meta's own guidance (docs):
  - a short noun phrase naming one concrete object, with a color or material attribute, for example "orange foam ball";
  - no verbs, no questions, no lists, no negation, no spatial phrases;
  - one concept per prompt.
- **Validation:**
  - Pydantic, with one repair retry if it fails.
  - If `proceed=false`, stop and show Wonder's reason along with the matching setup tips from Part 3.
- **Reasoning effort:** medium.

**Done when:** on 10 test clips, Wonder returns valid JSON every time, and its SAM prompts select the right object in at least 9.

## Part 7: Tracking with SAM 3.1

**What it does:** turns the video into per-frame masks of each tracked object and a mask of the calibration sheet, then into numeric features such as centroid, area and orientation.

**How to build it:**

- **Client:**
  - OpenAI Python SDK with `base_url="https://api.meta.ai/v1"`, using `client.responses.stream(model="sam-3.1", ...)` (docs).
  - Parse with `meta-sam-parser` (pip). Use version 0.1.0 or later for `build_object_prompt`; versions before 0.0.6 return no records when confidence is on (docs).
  - Use the parser's video-stream entry point, which mirrors `parseVideoStream`. Store the **raw event stream as JSONL** on the Volume, so the browser can replay it into `@meta-sam/parser` (Part 24).
- **Requests:**
  1. **Sheet:**
     - Build a median image from 15 frames spread across the clip; the median removes the moving object.
     - Send it as `input_image` with the prompt "white sheet of paper", or Wonder's phrase. Set `include_confidence: "true"` and keep the default lossless masks, because corner precision matters.
     - Repeat on the first, middle and last frames to detect camera motion (Part 8).
  2. **Each moving object, marker or pivot:**
     - Send `input_video` with the master's `file_id` and the concept prompt.
     - Set `include_confidence: "true"`, with streaming on.
     - Start with lossless masks. Switch video to `one_bit` only if payloads are slow, and check on one clip that the centroids don't change (docs: one_bit is several times smaller).
- **Reading the output** (docs):
  - Each frame arrives as one line.
  - Object IDs are stable within a response but sparse.
  - Frames with nothing visible emit no line.
  - Concept prompts return frames in ascending order.
  - Key everything by the `<Nf>` frame marker and the object ID, never by line position.
- **Choosing among several matches** (for example, two balls in the frame): pick the track with the largest total displacement; break ties by mean confidence.
- **Retry ladder when a prompt finds nothing or the wrong thing:**
  1. Ask Wonder to rephrase, showing it the empty result and a frame. At most two tries.
  2. Ask Wonder for a bounding box on a specific frame, in the master's pixel coordinates. Send it as a box prompt built with `build_object_prompt` and `propagation_direction: "both"` (docs).
  3. Fail with a clear message and filming tips.
- **Track stitching:** IDs aren't reliable across exits, entries and full occlusion (docs). When an object reappears under a new ID, join the tracks if its position is within three object radii of where the old track was heading.
- **Mask features** (NumPy and OpenCV moments on the decoded raster, offset to the box's origin):
  - area;
  - sub-pixel centroid (m10/m00, m01/m00);
  - orientation θ = ½·atan2(2μ11, μ20 − μ02), and eccentricity;
  - bounding box;
  - confidence `c`;
  - a flag when the mask touches the frame border (the object is partly out of frame, so exclude its centroid).
  - Motion blur stretches the mask, but a symmetric streak's centroid is the mid-exposure position. That constant time offset is absorbed by the fitted start time.
- **Caching:**
  - The key is sha256(master) + prompt + metadata.
  - SAM results vary slightly between runs (docs), so the cache also makes reruns reproducible.
- **Fallback:** sam-3.1 allows 3,000 requests per minute (docs), so a self-hosted SAM fallback isn't worth building. If the API is down, use replay mode and the seeded examples.
- **Output:** `tracks.parquet`, with columns object_key, frame, t, cx, cy, area, theta, conf and flags, plus the sheet mask.

**Done when:**
- a static object's centroid jitters by less than 0.3 px RMS;
- a ball is tracked continuously through a 600-frame drop;
- the mask overlay lines up with the object in the browser.

## Part 8: Calibration (scale, plane offset, gravity, timing)

**What it does:**
- Converts pixels to meters in the plane of motion.
- Corrects for the object moving in front of the sheet.
- Finds which way is down.
- Attaches an uncertainty to all of it.

**How to build it:**

- **Sheet corners:**
  1. Take the largest external contour of the sheet mask.
  2. Split it into four sides using an initial `approxPolyDP` quad.
  3. Fit a RANSAC line to each side's edge points and intersect the lines. This gives sub-pixel corners and tolerates rounded or curled corners.
  4. Refine with `cv2.cornerSubPix` on the grayscale median image when the corners are sharp.
  5. Order the corners top-left, top-right, bottom-right, bottom-left by angle around the centroid. Assign the longer pair of sides to 297 mm.
- **Paper size:**
  - A4 is 210 × 297 mm, aspect 1.414. US Letter is 215.9 × 279.4 mm, aspect 1.294.
  - When the camera faces the sheet nearly head-on, the image quad's aspect tells them apart. Preselect that in the confirm step and let the user override.
- **Homography:**
  - `H = cv2.getPerspectiveTransform(corners_px, corners_mm)`.
  - Sanity check: the rectified aspect must match the paper size within 2%. If not, fall back to `minAreaRect` and flag it.
- **Plane-offset correction.** The object's center moves a distance d in front of the sheet, toward the camera.
  - Estimate the camera-to-sheet distance: Z ≈ f_px × 0.297 m / (long side in px).
  - f_px ≈ image width in px × f35 / 36. Use metadata if present; otherwise default to 26 mm, the 1x main camera on most phones.
  - That f estimate is good to about 15%, which moves a 2% correction by about 0.3%.
  - Then world = c + (H(p) − c) × (1 − d/Z), where c = H(image center).
  - **Spin-down needs no correction:** angles in a plane parallel to the sheet are unaffected.
- **Gravity direction**, in order of preference:
  1. The data itself, for free fall and projectiles. The acceleration direction is a fitted parameter.
  2. A plumb reference: the pendulum at rest.
  3. Camera level: the image vertical mapped through H. This assumes the phone was leveled (Part 3).
  - Record which source was used.
- **Ramp angle:**
  - Segment the ramp ("wooden board") and fit a line to its top edge in world coordinates.
  - Cross-check against a PCA of the object's path. If they differ by more than 1°, flag it.
- **Camera motion:**
  - Compare the sheet corners in the first, middle and last frames.
  - Over 1 px: warn.
  - Over 3 px: stop and ask for a re-film with a tripod.
- **Uncertainty**, by Monte Carlo with 500 draws:
  - corners jittered by N(0, σ_c), with σ_c taken from the line-fit residuals (minimum 0.3 px);
  - d ± 0.5 cm;
  - Z ± 15%.
  - The result is σ_scale (relative) and the ramp angle's σ.
- **Timing:** taken from Part 5, including its jitter.
- **Output:** `calibration.json`, containing the corners, H and its inverse, mm per px at the center, the aspect check, the offset correction, gravity, the ramp angle, the uncertainty budget and the camera-motion reading. Also:
  - `calib_overlay.png`: the sheet outline plus a 5 cm world grid drawn through H⁻¹. It's shown in the UI and passed to Wonder and the reviewer.
  - `rectified.png`.

**Done when:**
- on a synthetic perspective render, the scale error is under 0.2%;
- on a real clip with a tape measure in the plane, 50 cm reads 500 ± 2 mm.

## Part 9: Trajectory extraction and cleaning

**What it does:** turns tracks into clean physical time series (x, y, angle) in meters, radians and real seconds, with noise estimates, events and quality flags.

**How to build it:**

- **World transform:** convert each centroid through H, apply the offset correction, then rotate into the physics frame (x horizontal, y up). Store positions in meters.
- **Angles:**
  - **Pendulum:** θ = atan2(x_bob − x_pivot, −(y_bob − y_pivot)).
    - If no pivot was segmented, find it by fitting a circle to the bob's path: a Kasa algebraic fit, refined with geometric least squares.
    - L is the circle's radius.
    - Effective length for a spherical bob of radius r: L_eff = L + (2/5)r²/L.
  - **Spinner:** φ = atan2(marker − hub), then `np.unwrap`.
    - If any step exceeds 120° per frame, flag possible aliasing (spin too fast for the frame rate).
    - Check that the spin direction is consistent.
- **Cleaning:** drop frames with any of the following:
  - confidence under 0.5;
  - a mask touching the frame border;
  - |area − rolling median| over 4×MAD;
  - a centroid jump more than five times the median step.
  - Keep the gaps; never interpolate data that will be fitted.
- **Noise:** robust standard deviation (MAD × 1.4826) of residuals from a local cubic Savitzky–Golay fit with a window of about 15 frames.
- **Events:**
  - **Release:** speed more than 3σ above noise for five frames in a row.
  - **Impacts (bounces):** the vertical velocity flips sign near a minimum of y. Split the motion into flights there, and drop two frames on each side of each impact.
  - **Stop:** speed under threshold for more than 0.5 s.
- **Out-of-plane check:**
  - The mask area scales as 1/Z². If the area varies by more than 5% and correlates with the motion (|r| over 0.5), flag "possible out-of-plane motion".
  - Estimate ΔZ/Z ≈ ΔA/(2A) and add it to the uncertainty budget.
- **Output:**
  - `trajectory.parquet`: t, x, y, θ or φ, weight, flags, segment_id.
  - `traj_summary.json`: point count, duration, noise σ, events, ranges, out-of-plane metric, and about 60 evenly spaced sample points for model context.
- **Velocities:** Savitzky–Golay derivatives, for display and diagnostics only.

**Done when:** synthetic noisy trajectories round-trip within tolerance, and the noise estimate lands within 20% of the true noise.

---

# C. Physics reasoning

## Part 10: The physics model library

**What it does:** gives every supported system a set of competing, physically distinct models, with equations, parameters and plain-language mechanisms, ready to fit and to simulate.

**How to build it:**

- **The interface** shared by library models and Wonder-written models (`physics/library/base.py`):
  ```python
  class ParamSpec(BaseModel):
      init: float; lower: float; upper: float; unit: str
      kind: Literal["physical", "initial_condition", "nuisance"]

  class ModelDef:
      id: str; name: str; system: str
      mechanism: str              # one sentence a student understands
      equations_latex: str
      observables: list[str]      # ["y"], ["x", "y"], ["theta"], ["phi"]
      state: list[str]
      params: dict[str, ParamSpec]
      def initial_state(self, p) -> np.ndarray: ...
      def rhs(self, t, s, p) -> np.ndarray: ...            # or None if closed form
      def closed_form(self, t, p) -> np.ndarray | None: ...
      def observe(self, s) -> np.ndarray: ...
      signature: str              # what this model predicts that its rivals don't
  ```
- **Library contents:**

| System | Models |
|---|---|
| Free fall / projectile | **FF0** constant g (closed form). **FF1** linear drag: v′ = −g ŷ − (b/m)v. **FF2** quadratic drag: v′ = −g ŷ − (k/m)\|v\|v; also reports terminal velocity. |
| Pendulum (θ) | **P0** small-angle SHM: θ = A cos(ωt + φ). **P1** nonlinear: θ″ = −(g/L_eff) sin θ. **P2** P1 plus viscous damping −γθ′. **P3** P1 plus quadratic air drag −κ\|θ′\|θ′. **P4** P1 plus Coulomb pivot friction −μ_p tanh(θ′/ε). |
| Ramp (s along the incline) | **R0** frictionless slide: s″ = g sin α. **R1** kinetic friction: s″ = g(sin α − μ_k cos α). **R2** rolling without slipping: s″ = g sin α/(1 + β), with β = 2/5 for a solid sphere, 2/3 for a hollow sphere, 1/2 for a solid cylinder. **R3** R2 plus rolling resistance c_rr. |
| Bounce | Fit each flight as a parabola to get the speeds just before and after each impact. **B0** constant restitution e. **B1** speed-dependent e = e0 − e1\|v\|. **B2** flights with quadratic drag plus constant e. |
| Spin-down (φ) | **S0** viscous: ω′ = −bω (exponential). **S1** Coulomb: ω′ = −c (linear). **S2** both: ω′ = −c − bω. **S3** Coulomb plus air: ω′ = −c − kω². All have closed forms; ω clamps at 0. |
| Stretch: spring and mass | y″ = −(k/m)(y − y_eq) − γy′ |

- **Physical parameters are fit as physics, not as curve shapes.**
  - **Pendulum:** fit g with L_eff fixed from the video measurement. If the user entered a tape-measured L, report both g(video L) and g(tape L).
  - **Ramp:** fix g at the local value (Part 25) and fit μ_k or β.
- **Each model ships with:** a mechanism sentence, LaTeX equations, a signature string, data-driven initial-guess heuristics, and a TypeScript twin for the what-if simulator (Part 18).

**Done when:** for each system, 50 synthetic datasets at realistic noise:
- recover the true parameters within 2σ, and
- select the true model at least 90% of the time.

## Part 11: Fitting harness and the Modal Sandbox

**What it does:** runs model fits, including untrusted Wonder-written model code, in an isolated container, and returns parameters, uncertainty, diagnostics and plots.

**How to build it:**

- **Sandbox:**
  - Created with `modal.Sandbox.create(app=app, image=fit_image, block_network=True, timeout=600, idle_timeout=120, cpu=2, memory=2048)` (docs).
  - `fit_image` contains Python 3.12, numpy, scipy, pandas, matplotlib, sympy, and PyBullet for the export replay check (Part 19).
  - One sandbox per analysis job, reused for every fit in that job, and terminated at the end.
- **I/O:**
  - Write `data.json`, `job.json` and `models/<id>.py` with `sb.filesystem.write_text`.
  - Run `sb.exec("python", "/harness/run_fit.py", "/work/job.json", timeout=120)`.
  - Read `result.json` and the PNG plots back with `sb.filesystem.copy_to_local` (docs).
- **Before anything runs, check Wonder's code:**
  - Parse it with `ast` and allow only `math`, `numpy` and `scipy.special` imports.
  - Reject `open`, `exec`, `eval`, `__import__`, and attribute access to `os` and `sys`.
  - Check the function signatures.
  - Dry-run `rhs` on a sample state to catch shape and NaN errors.
  - Return clear errors to Wonder; it gets two attempts to fix them.
- **Fitting:**
  - Scale parameters to order 1.
  - Initial guesses come from heuristics: a polynomial fit for FF0, the FFT peak and zero crossings for the pendulum, a linear fit of ω for the spinner.
  - `scipy.optimize.least_squares(method="trf", loss="soft_l1", f_scale=2.0, x_scale="jac", bounds=...)`. Residuals are (observed − predicted)/σ, weighted by confidence.
  - Nonlinear models get eight multi-start runs from Sobol points around the heuristic guess; keep the best cost.
  - Use the closed form when available. Otherwise use `solve_ivp(method="DOP853", rtol=1e-9, atol=1e-11, t_eval=t)`, falling back to `LSODA` if it fails.
  - Cap wall time at 30 s per model.
- **Uncertainty:**
  - Jacobian covariance: s²(JᵀJ)⁻¹.
  - Block residual bootstrap: blocks of about 10 frames, which preserves autocorrelation, with 200 resamples and a fixed seed. The headline interval is the 2.5–97.5 percentile range.
- **Diagnostics:**
  - RMSE, reduced χ², lag-1 autocorrelation of the residuals, Durbin–Watson statistic, and a runs-test p-value.
  - A "systematic residual" flag when a low-order sinusoid or polynomial explains more than 30% of the residual variance.
- **Plots** (matplotlib, Agg backend, 1000×600 PNG):
  - data plus fit, with the held-out window shaded;
  - residuals against time;
  - for spin-down, ω(t).
- **Output caps:** `result.json` at most 5 MB; at most four plots per model.

**Done when:** a five-model comparison on a 600-point series finishes in under 20 s inside the sandbox, and malicious test code (a file read or a network call) is rejected or fails safely.

## Part 12: Model selection and uncertainty

**What it does:** decides which model wins by how well it predicts unseen data, decides whether the win is real, and builds the final numbers with an honest uncertainty budget.

**How to build it:**

- **Split:**
  - Chronological: fit parameters and initial conditions on the first 65% of the motion window, then integrate forward through the last 35%. The test is a pure forecast.
  - For bounces, fit on the first k flights and predict the next one.
- **Score:**
  - Held-out RMSE, in meters or radians.
  - Also shown: NRMSE (RMSE as a fraction of the motion's range) and the time to divergence (the first time the error exceeds 3σ_noise).
- **Is the win real?**
  - Run a paired block bootstrap of the held-out errors.
  - Model A beats model B only if the 95% interval for RMSE_B − RMSE_A excludes zero.
- **Robustness:** repeat with splits at 50% and 80%. The winner must win at least two of the three splits; otherwise the result is "inconclusive".
- **Parsimony:** if the top two can't be separated, prefer the model with fewer physical parameters. Mark confidence medium or low, and say "the data can't distinguish A from B".
- **Extra information, for display only:** AIC and BIC on the full data.
- **Final numbers:**
  - Refit the winner on all the data.
  - Combine uncertainty by re-running the final fit 50 times with perturbed calibration (scale, d, Z, angle) and timing.
  - σ_total = √(σ_stat² + σ_calib² + σ_timing²), plus a model-choice term: the spread of the parameter across models whose held-out error is statistically tied.
- **What depends on what:**
  - g from free fall scales with the calibration scale and with time⁻².
  - g from the pendulum scales with L.
  - μ_k and β depend on the ramp angle, not on scale.
  - e and the spinner's parameters depend only on timing.

**Done when:** on synthetic data with known truth, the 95% intervals contain the truth 90–98% of the time.

## Part 13: The physicist agent loop (Muse Wonder)

**What it does:** Wonder plays the scientist. It proposes competing explanations, predicts in advance what would refute each one, calls the harness to test them, reads the evidence, rejects what fails, and writes conclusions in plain language. Every number it cites comes from the harness.

**How to build it:**

- **The orchestrator runs a fixed state machine:**
  - SCENE (Part 6) → MEASURE (Parts 7–9, automatic) → HYPOTHESIZE → FIT → ANALYZE → REVISE (at most 2 rounds) → CONCLUDE → REVIEW (Part 14) → RESPOND (at most 2 rounds) → FINAL.
  - Wonder acts only through tools.
- **Context at HYPOTHESIZE:**
  - the system prompt (Appendix B) and the library catalog, first, so prompt caching applies ($0.20 per 1M cached input, docs);
  - the scene analysis, including its initial hypotheses;
  - a summary of the **training window only**, with about 60 sample points and a plot;
  - the calibration summary;
  - the user's hint.
- **Tools** (function calling on the Responses API):
  1. `list_library_models(system)`: catalog entries.
  2. `get_training_data(t_start, t_end, max_points ≤ 400)`: numeric arrays. Training window only.
  3. `view_frames(times[], overlay: none|mask|trail)`: up to six frames. Returned as images, or attached in a follow-up message (see Part 1, check 3).
  4. `propose_models(models[])`: for each model, its ID, name, mechanism, LaTeX, either a library reference or code with a parameter spec, its prior rank, and its refuting signature. Returns validation results.
  5. `run_fits(model_ids[], split: "selection"|"final")`: the comparison table. For each model: parameters ± CI, training RMSE, held-out RMSE ± CI, diagnostics, and plot IDs.
  6. `view_plot(plot_id)`: the image.
  7. `submit_conclusion(...)`: the winner, quantities with CIs, rejected models with reasons and evidence, confidence, caveats, and the narrative. All in Markdown, with the numbers validated against the harness output.
- **Rules in the system prompt:**
  - Propose three to five models with different mechanisms before any fit.
  - Name your favorite in advance, and say what result would refute each model. This is recorded verbatim in the notebook.
  - Every rejection must cite a held-out number or a residual pattern.
  - Say "can't tell" when the models are tied.
  - Don't add parameters without a mechanism.
  - Keep narratives short and readable by a student.
- **Validation by the orchestrator:**
  - Every number in `submit_conclusion` must match a harness result within rounding. Otherwise the call is rejected and the mismatch is sent back to Wonder.
  - Wonder can't introduce a number the harness didn't produce.
- **Budgets:**
  - At most 8 tool calls per phase, 2 revision rounds, and 4 custom models.
  - Reasoning effort is high for HYPOTHESIZE and CONCLUDE, and medium or low elsewhere.
  - `max_output_tokens` of 8,000 or more, because reasoning shares the output budget (docs).
  - Always stream.
- **Failures:**
  - Invalid tool arguments: return a validation error, which counts against the budget.
  - 429 or 5xx: back off (Part 23).
  - Wonder still failing: **library-only mode**. Fit all library models for the system, select by held-out error, and write the narrative from templates. The user always gets a result, and the notebook says Wonder was unavailable.

**Done when:**
- On the spin-down clip, the notebook shows Wonder's stated favorite, the fits, and a rejection that cites held-out numbers.
- On 10 clips, the loop finishes in under 3 minutes each, at the Wonder cost measured in Part 1 or less.

## Part 14: The reviewer (Muse Spark Open on a Modal Shared Endpoint)

**What it does:** a second, independent model acts as a skeptical referee. It reads the claimed law and the evidence, including the plots, and challenges anything the evidence doesn't support, before Ghostline reports the result.

**How to build it:**

- **Client:**
  - `OpenAI(base_url=f"{REVIEWER_BASE_URL}/v1", api_key=REVIEWER_API_KEY)`, using Chat Completions with `model=REVIEWER_MODEL`.
  - Pass `extra_headers={"Modal-Routing-Affinity-Key": experiment_id}`. This keeps the conversation's prompt prefix in one container's cache across rounds (docs).
  - Cached input is $0.15 per 1M versus $1.25.
- **Model:** Muse Spark Open is the open-weight release of Muse Spark, an 816B-parameter mixture-of-experts model with 42B active parameters (docs). It accepts text and images on Modal.
- **Input:**
  - the conclusion JSON;
  - the full comparison table (held-out RMSE ± CI, parameters ± CI, AIC/BIC, diagnostics);
  - the data-quality flags;
  - the calibration and timing uncertainty budget;
  - two images as `image_url` data URLs: the fit plot and the residual plot.
  - It never sees Wonder's private reasoning, only the artifacts, the way a real referee would.
- **Checklist in the prompt:**
  1. Is the winner's held-out advantage larger than its uncertainty?
  2. Do the residuals show structure, meaning missing physics?
  3. Are the parameters physically plausible and away from their bounds?
  4. Are the intervals believable, given the calibration?
  5. Do any quality flags (out-of-plane motion, camera motion, re-timed video) undercut the claim?
  6. Does the narrative overstate the numbers?
  7. Is there a simpler explanation that wasn't tested?
- **Output:** the `Review` JSON from Appendix C, with a verdict (accept, accept_with_caveats or challenge) and a list of issues. Use `response_format` with a JSON schema if the endpoint supports it (Part 1, check 6). Otherwise parse, with one repair retry.
- **The debate:**
  1. On a challenge with any major issue, the issues go to Wonder as a new turn, with tools enabled. Wonder can run a model, refit, lower its confidence, or rebut with numbers.
  2. The reviewer then gets round 2, with the history.
  3. Final status:
     - "Accepted",
     - "Accepted with caveats", or
     - "Contested", if the reviewer still challenges after two rounds. Both sides are shown, never hidden.
- **If the endpoint is down:** the analysis still completes, with the status "Not reviewed (reviewer unavailable)".

**Done when:** on two planted bad cases, the reviewer challenges with the specific correct issue:
- a forced overfit (a model with one extra free parameter that wins on training data), and
- a re-timed clip.

## Part 15: Confidence and "can't tell"

**What it does:** turns the evidence into a confidence level the user can trust, with reasons, and says plainly when Ghostline doesn't know.

**How to build it:** deterministic rules in `confidence.py`.

- **High:** all of the following.
  - The winner beats the runner-up in all three splits.
  - No systematic-residual flag.
  - No major quality flags.
  - The reviewer accepted.
- **Medium:** any of the following.
  - The winner beats the runner-up in two of three splits.
  - There are minor flags.
  - The reviewer accepted with caveats.
- **Low:** any of the following.
  - The models are tied.
  - There's a major flag: out-of-plane motion, camera motion, an uncertain time base, under two periods of motion, or under 0.5 s of motion.
  - The verdict is contested.

Low confidence shows a "Ghostline can't tell yet" card with specific fixes, such as "film a longer clip", "increase the amplitude", or "use slow-mo". Every quantity has an expandable uncertainty table: statistical, calibration scale, plane offset, timing and model choice.

**Done when:** the three deliberately bad benchmark clips (Part 25) all come out low confidence, with the right reason.

---

# D. Proof and exploration

## Part 16: Prove it: ghost prediction and overlay

**What it does:** on a new trial the model has never seen, Ghostline draws a translucent ghost where the fitted law says the object will be, and measures how far reality drifts from it.

**How to build it:**

- **Flow:** **Prove it** → upload trial 2, using the same setup with any starting conditions → ingest (Part 5) → tracking and a fresh calibration (Parts 7–8; the sheet is re-detected) → trajectory (Part 9).
- **Prediction:**
  - Freeze the physical parameters from trial 1's final fit.
  - Fit **only the initial conditions** to trial 2's launch window: the first 10% of the motion or 0.25 s, whichever is longer. That means position and velocity, θ0 and ω0, or φ0 and ω0.
  - Everything after the launch window is a pure prediction.
  - Uncertainty: 200 Monte Carlo draws over the parameter CIs and the initial-condition uncertainty give a spread of predicted positions for each frame.
- **To pixels:**
  - world → remove the offset correction → H⁻¹ of trial 2 → pixel coordinates.
- **The ghost sprite:**
  - Cut the object out of the trial-2 frame with its sharpest, highest-confidence mask.
  - Feather the alpha with a 2 px Gaussian blur.
  - Blend the color 50% toward cyan, at 0.55 opacity, with a soft glow (dilated, blurred alpha at 0.25).
  - Rotate the sprite by the predicted angle for spinners.
  - For pendulums, draw a faint "ghost string" from the pivot.
- **Divergence:**
  - Per-frame error e_t = |actual − predicted|, in mm.
  - Threshold τ = max(3σ_noise, half the object's radius).
  - Onset is the first time e > τ for five or more frames in a row.
  - The ghost's outline goes green when e < τ, amber up to 2τ, and red beyond that.
  - On-screen text shows "Δ 4 mm" and "predicting 2.8 s ahead", plus a "launch window" label over the first frames.
- **Server-side render** (Modal Function, `cpu=4`):
  1. Decode trial 2's master with PyAV.
  2. Draw on each frame: the real trail (thin white, last 0.5 s), the predicted path (dotted cyan), the ghost sprite and halo, and the on-screen text.
  3. Encode H.264 with the original timestamps and faststart → `ghost_v1.mp4`.
  4. Slow-mo clips also get a 30 fps slow-playback version.
  5. Write `ghost_track.json` with predicted px, actual px and error for every frame.
- **The error chart:**
  - e(t), with the τ line and the onset marker.
  - Its cursor stays in sync with video time.
- **Stretch: interactive overlay in the browser.**
  - `@meta-sam/react`'s `Video`/`useMediaPlayer` decodes frames into a canvas with frame-accurate indexes (docs).
  - Draw the ghost from `ghost_track.json` per frame index.
  - Add toggles for the ghost, the uncertainty halo and the SAM mask, plus frame-by-frame scrubbing.

**Done when:**
- on a new pendulum trial, the ghost stays within τ for at least three periods;
- a 10 s clip renders in under 60 s.

## Part 17: Divergence diagnosis and revision

**What it does:** when the ghost drifts, Wonder looks at where and how it drifted, names the missing physics, and proposes a revised model. If the revision predicts better, Ghostline renders ghost v2.

**How to build it:**

- **Trigger:** a divergence onset, or the user tapping **Why did it drift?**
- **Input to Wonder:**
  - the error-vs-time plot;
  - four rendered frames around the onset, with ghost and object both visible;
  - the model, its parameters, and the trial-1 summary.
  - It returns a diagnosis (for example, "the ghost lags late in the swing, so damping is underestimated"), a revised model (from the library or custom code), and the signature it predicts.
- **Testing the revision:**
  - Fit the revised model on trial 1 only, with the Part 12 protocol.
  - Compare its trial-2 forecast with v1's. After the launch window, trial 2 is held-out data for both models.
  - Adopt v2 only if it's better beyond noise. Otherwise record "the revision didn't help".
- **Review:** one reviewer round on the revision.
- **No overfitting to trial 2:** physical parameters never come from trial 2.
- **UI:**
  - A "Ghost v1 / v2" toggle.
  - Both error curves on one chart.
  - Notebook entries for the diagnosis and the test.

**Done when:** a deliberately incomplete v1 (an undamped pendulum) gets a diagnosis that names damping, and a v2 ghost that tracks noticeably longer.

## Part 18: What-if simulator

**What it does:** makes the fitted law a live, editable simulation. Change gravity, angle, length, mass or air, and watch the user's own object move in the user's own scene.

**How to build it:**

- **Client-side simulation:**
  - `web/src/lib/sim/models/*.ts` are twins of every library model, with the same equations and parameter names.
  - A fixed-step RK4 integrator (dt = 1/240 s) with event handling for bounces and the ω → 0 clamp.
  - **Parity tests:** Python generates fixture trajectories, and `vitest` checks the TypeScript output against them within 1e-6 relative.
- **Custom (Wonder-written) models:** `POST /simulate` runs them in the sandbox, with a spinner in the UI.
- **Controls** (sliders in physical units, with presets):
  - **Gravity:** Earth 9.81, Moon 1.62, Mars 3.71, Jupiter 24.79 m/s², or custom.
  - **Air density:** Earth ×1, Mars ×0.01, vacuum ×0. Drag coefficients scale with density.
  - **Pendulum:** length, amplitude, and bob mass. Damping was fitted as b/m, so a heavier bob of the same size damps more slowly. The UI points out that mass doesn't change the period.
  - **Ramp:** angle, μ_k, and shape (block, solid ball, hollow ball, solid cylinder).
  - **Bounce:** drop height and e.
  - **Spinner:** initial spin. Bearing friction scales with g, because the load is the spinner's weight; the UI states that assumption.
- **Rendering:**
  - A canvas over a still frame from trial 1.
  - The object sprite is mapped through H⁻¹, so "your pendulum, on Mars" plays in your own scene.
  - Two ghosts side by side: the fitted model and the what-if.
  - A time-series chart, plus derived readouts: period, time to stop, bounce height, terminal velocity.
- **Explanation:**
  - Instant explanations from formulas, for example "T ∝ 1/√g, so the period is 1.63× longer".
  - An optional **Explain** button sends the computed numbers to Wonder for two or three sentences, and the answer is cached.
- **Guardrail:** an extrapolation warning when a slider leaves the measured range, for example "friction was measured at 15–25°".

**Done when:**
- the animation holds 60 fps on a mid-range phone;
- the parity tests pass;
- the Mars pendulum's period is exactly √(9.81/3.71) ≈ 1.63× Earth's.

## Part 19: Export to simulator

**What it does:** turns the measured parameters into a file a robotics simulator can load. A team films a real part sliding, bouncing or swinging, and gets its friction, restitution and damping, plus ranges to randomize over during training. A replay check then runs the same experiment inside the simulator, to show that the exported numbers reproduce the real motion.

**How to build it:**

- **What gets exported:** the winning model's parameters from the final fit (Part 12). Each one comes with its value, its 95% interval, its unit, and the conditions it was measured under.

| Measured in Ghostline | Ghostline JSON field | PyBullet |
|---|---|---|
| Sliding friction μ_k (ramp, R1) | `friction.kinetic` | `lateralFriction` |
| Rolling resistance c_rr (ramp, R3) | `friction.rolling` | `rollingFriction` = c_rr × ball radius, because Bullet's rolling friction has units of length |
| Restitution e (bounce, B0) | `restitution` | `restitution` |
| Viscous decay rate λ = b/m (free fall, FF1) | `damping.linear_rate` | `linearDamping` = 1 − e^(−λ), because Bullet scales velocity by (1 − damping)^dt every step |
| Joint viscous damping γ (pendulum P2; spinner S0, S2) | `joint.damping` | `jointDamping` = γ × I |
| Joint Coulomb friction (pendulum P4; spinner S1, S2) | `joint.coulomb_torque` | A velocity motor with target 0 and maximum force c × I, the usual way to model joint friction in PyBullet |
| Quadratic drag (FF2, P3, S3) | `drag.quadratic` | No matching field. The snippet applies the force every step. |

- **Speed-dependent bounce (B1):** PyBullet only takes a constant restitution, so it gets e at the median measured impact speed. The JSON keeps both coefficients.
- **Bullet multiplies the two bodies' values.** Bullet combines the friction and restitution of two touching bodies by multiplying them. The measurement describes a pair (a wooden block on a painted board), so the snippet sets the object's value to the measurement and the surface's value to 1, with a comment saying why.
- **No hidden damping.** PyBullet bodies come with small default linear and angular damping, and revolute joints start with a velocity motor that holds them still. The snippet sets all of these explicitly, so the simulation loses energy only where the measurement says it should.
- **Mass and inertia.** Ghostline measures rates (λ, γ, c), not absolute coefficients.
  - The export dialog asks for the object's mass in grams; a kitchen scale is enough.
  - For the pendulum, I = m(L² + (2/5)r²), from the measured length and bob radius.
  - A spinner's mass distribution can't be seen on video, so its values stay per unit of inertia (rad/s² and 1/s), and only the JSON format is offered, unless the user enters I.
- **Randomization ranges:** each 95% interval becomes a `randomization` low–high pair. The export notes that this covers measurement uncertainty only, and that part-to-part variation needs wider ranges.
- **Where the numbers apply:** the surface pair from the scene analysis, and the speed, angle and amplitude ranges that were measured. Using the numbers outside those ranges is extrapolation, and the file says so.
- **Formats:**
  1. **Ghostline JSON**, the canonical format (`SimExport` in Appendix C).
  2. **PyBullet snippet**: a short Python script of `changeDynamics` and `setJointMotorControl2` calls, ready to paste.
  - Stretch: MuJoCo MJCF (`<geom friction="...">`, `<joint damping="..." frictionloss="...">`) and a USD physics material for Isaac Sim (`physics:dynamicFriction`, `physics:restitution`). MuJoCo has no restitution field; bounciness comes from its contact `solref` settings, so that format needs the replay check to tune it.
- **Replay check** (`export/verify_pybullet.py`, run in the Modal Sandbox with PyBullet in the fit image):
  1. Rebuild the experiment from the measured geometry: a box on a plane tilted to the measured angle, a sphere dropped from the measured height, or a pendulum URDF with a revolute joint and the measured length.
  2. Apply the exported parameters and simulate at 240 Hz.
  3. Compare against the measurement: the ramp's acceleration, the ratio of successive bounce heights, or the pendulum's period and decay.
  4. Pass if every simulated value falls inside the measured 95% interval. The UI shows "Reproduced in PyBullet" with both numbers, or the mismatch. Simulators lose energy in contacts differently from the real world, and this check is what catches it.
- **UI:** an **Export to simulator** button on the Result tab, shown only when the winning model has exportable parameters. It opens a dialog with the format choice, the mass field, the generated snippet with copy and download buttons, and the replay-check result.
- **API:** `GET /api/experiments/{id}/export` with `format` and `mass_g`, and `POST /api/experiments/{id}/export/verify`, which runs the replay check as a job (Part 21).
- **Effort:** about 3–4 hours for the JSON, the PyBullet snippet, and the replay check on the ramp and bounce. MuJoCo and USD are stretch goals.

**Done when:** for the benchmark's ramp-block and bounce clips, the exported PyBullet parameters reproduce the measured acceleration and bounce ratio inside their 95% intervals, and the snippet runs unedited after a fresh `pip install pybullet`.

## Part 20: The lab notebook

**What it does:** records every step of the reasoning as an append-only timeline that a student can read, question, and re-run. It's the evidence that the scientific method actually happened.

**How to build it:**

- **The event row:**
  - `id, experiment_id, seq, ts, phase`
  - `actor`: wonder, spark_open, harness, sam, system or user
  - `kind`: observation, hypothesis, prediction, fit, rejection, acceptance, review, response, revision, ghost, whatif or warning
  - `title, body_md` (Markdown with LaTeX)
  - `data` (JSON), `attachments`, `refs`
- **The orchestrator writes events, not the model:**
  - Every harness result becomes an event whose numbers come straight from the harness.
  - Wonder's narrative fields are attached as body text.
  - Events are never edited. Re-runs append new events.
- **UI:**
  - A vertical timeline, colored by actor, filterable between "key decisions" and "everything".
  - KaTeX for equations.
  - Expandable raw details: code, parameters, JSON.
  - Pinned key moments:
    - **Prediction made before data** (Wonder's favorite and its refuting signature),
    - **Hypothesis rejected** (with the held-out numbers),
    - **Reviewer challenge**,
    - **Ghost verdict**.
- **Re-running a step:**
  - Every fit event has a **Re-run** button with an editable split percentage and initial guesses.
  - Stretch: edit the model's code. It goes through the same AST check and sandbox.
- **Export (stretch):** a Markdown lab report download; the browser's print to PDF handles the rest.

**Done when:** someone who never saw the UI can follow the spin-down story from the notebook alone.

---

# E. Platform

## Part 21: Back-end API and job orchestration

**What it does:** the FastAPI service the web app talks to, plus the background jobs that run the pipeline.

**How to build it:**

- **Routes** (all under `/api`; write routes require the `X-Ghostline-Key` passcode header):

| Method and path | Purpose |
|---|---|
| `POST /experiments` | Create an experiment. Takes the title, system hint, description, declared capture mode and fps, plane offset, paper size and measured L. |
| `POST /uploads` | Start a chunked upload: experiment, role (fit or prove), filename, size. Returns `upload_id`. |
| `PUT /uploads/{id}/chunks/{n}` | A raw chunk of at most 8 MiB. Idempotent. |
| `POST /uploads/{id}/complete` | Assemble the file and spawn `ingest_trial`. Returns `trial_id`. |
| `POST /trials/{id}/trim` | Set the trim start and end (defaults to the auto-suggestion). |
| `POST /experiments/{id}/analyze` | Spawn `analyze_experiment`. |
| `POST /experiments/{id}/prove` | Spawn `prove_trial` for trial 2. |
| `POST /experiments/{id}/revise` | Spawn the divergence diagnosis and revision. |
| `POST /experiments/{id}/simulate` | Simulate a custom model (synchronous, at most 20 s). |
| `GET /experiments/{id}/export` | Download simulator parameters. Query: `format` (`json` or `pybullet`) and `mass_g` (Part 19). |
| `POST /experiments/{id}/export/verify` | Spawn the PyBullet replay check (Part 19). |
| `POST /experiments/{id}/models/{mid}/rerun` | Re-run a fit with edited settings. |
| `GET /experiments/{id}` | Full state: trials, models, fits, review, conclusion, ghosts, costs (admin only). |
| `GET /experiments/{id}/events?after=seq` | Notebook and progress events, for polling. |
| `GET /experiments?examples=true` | Gallery listing. |
| `GET /media/{exp}/{path}` | Media from the Volume, with HTTP Range support (206 Partial Content). Safari won't play video without it. Long-lived cache headers for content-hashed names. |
| `GET /benchmark`, `PUT /benchmark/{exp}` | Benchmark table and ground-truth entry. |
| `GET /health` | Database, Volume, and cached Meta API reachability. |

- **Jobs:**
  - Modal Functions started with `.spawn()`.
  - Each has a `jobs` row: kind, status, phase, progress 0–1, heartbeat and error.
  - Only one active job per experiment, enforced with a Postgres advisory lock.
  - Workers write events and a heartbeat every 5 s. A job without a heartbeat for 60 s is marked failed, with a **Retry** button.
  - Phases are idempotent where possible: cached SAM and model calls make retries cheap.
- **Function sizing:**

| Function | Resources |
|---|---|
| `api` | `cpu=0.5`, `memory=512`, `max_containers=5` |
| `ingest_trial` | `cpu=4`, `memory=8192`, `timeout=900` |
| `analyze_experiment` | `cpu=2`, `memory=4096`, `timeout=1800` |
| `prove_trial` | `cpu=4`, `memory=8192`, `timeout=1200` |

- **Media serving:**
  - Starlette's `FileResponse` handles Range in recent versions. **Verify**, and implement 206 by hand if not.
  - Modal web request bodies can be up to 4 GiB and responses are unlimited (docs).
- **Validation:** Pydantic models for every request and response. FastAPI's OpenAPI spec is used to generate the TypeScript types.

**Done when:** the full flow (create → upload → trim → analyze → prove → simulate) works through the API, with `curl` alone.

## Part 22: Data model and storage

**What it does:** keeps state in Postgres and files on a Modal Volume, with clear ownership.

**How to build it:**

- **Postgres tables** (SQL migrations in `backend/migrations/`):
  - `experiments`: id, created_at, title, system_type, status, declared_capture, setup (offset, paper, L), scene, winner_model_id, conclusion, confidence, review_status, is_example, is_benchmark, visibility, cost_usd, error.
  - `trials`: id, experiment_id, role, status, paths, meta_file_ids, probe, timing, trim, n_frames, calibration, created_at.
  - `tracks`: trial_id, object_key, sam_prompt, cache_key, parquet_path, stats.
  - `trajectories`: trial_id, parquet_path, summary.
  - `models`: id, experiment_id, round, source (library or wonder), library_ref, code, name, mechanism, equations_latex, params_spec, prior_rank, refuting_signature, status, rejection_reason.
  - `fits`: id, model_id, trial_id, split, params (value, ci_low, ci_high, unit), train_rmse, heldout_rmse, heldout_ci, aic, bic, diagnostics, plot_paths.
  - `reviews`: id, experiment_id, round, verdict, issues, summary, response.
  - `ghost_renders`: id, experiment_id, trial_id, model_id, version, video_path, track_path, divergence_onset_s, max_error_mm.
  - `events`: bigserial id, experiment_id, seq, ts, phase, actor, kind, title, body_md, data, attachments, refs. Unique on (experiment_id, seq).
  - `llm_calls`: experiment_id, phase, provider, model, input_tokens, cached_tokens, output_tokens, frames, cost_usd, latency_ms, cache_hit, request_hash.
  - `jobs`: id, experiment_id, kind, status, phase, progress, started_at, heartbeat_at, error.
  - `benchmark`: experiment_id, system, quantity, truth_value, truth_sigma, truth_method, notes.
  - `uploads`: id, experiment_id, role, size, chunk_size, received_chunks, status.
- **Volume layout** (`ghostline-data`):
  ```
  /experiments/{exp}/trials/{trial}/original.*, master.mp4, playback_720.mp4, wonder_480.mp4,
      audio.wav, poster.jpg, timestamps.npy, probe.json, timing.json,
      sam/{object_key}.events.jsonl, tracks.parquet, trajectory.parquet,
      calib/overlay.png, calib/rectified.png, sprite.png
  /experiments/{exp}/fits/{fit}/fit.png, residuals.png
  /experiments/{exp}/ghost/v{n}.mp4, v{n}_track.json, v{n}_slow.mp4
  /cache/llm/{hash}.json        /cache/sam/{hash}.events.jsonl
  /uploads/{upload_id}/{n}.part
  ```
- **Volume rules:**
  - Workers call `volume.commit()` after each phase.
  - The API calls `volume.reload()` before serving a path it hasn't seen yet.
  - Every file name that changes is versioned (`v2`, content hash), so caching is safe.

**Done when:** deleting an experiment removes its rows, its Volume folder, and its Meta Files API uploads.

## Part 23: Model clients, record/replay and the budget guard

**What it does:** one well-behaved layer for every model call: retries, structured-output validation, replay for tests and demos, and a meter that stops spending before it surprises you.

**How to build it:**

- **`llm/meta.py`:**
  - The OpenAI SDK pointed at `https://api.meta.ai/v1`.
  - Helpers:
    - `upload_file(path, expires_s=604800)`
    - `respond(model, input, tools, text_format, reasoning_effort, max_output_tokens, stream=True)`
    - `sam_video(file_id, prompt, metadata)`
    - `sam_image(image_bytes, prompt)`
- **`llm/reviewer.py`:** the Spark Open client with the affinity header.
- **Retries:**
  - On 429, 5xx and timeouts: exponential backoff starting at 500 ms, doubling, with jitter, honoring `Retry-After`. At most four attempts.
  - **Never retry a 400 unchanged.** Meta's docs say a 400 is permanent.
  - Log `x-ratelimit-remaining-*` (docs).
- **Structured output:** validate with Pydantic. On failure, make one repair call that includes the validation error.
- **Record and replay:**
  - `LLM_MODE` is `live`, `record` or `replay`.
  - Each call is stored under a hash of the normalized request (model, input, tools, parameters, secrets removed).
  - CI, tests and the demo fallback use replay.
- **The cost meter:**
  - Computes cost from the usage numbers: Wonder token prices, Spark Open token prices, and SAM per frame using `usage.video_frames_processed` (docs).
  - Writes `llm_calls` and adds to `experiments.cost_usd`.
- **Budget guard:**
  - `BUDGET_USD_PER_EXPERIMENT`, default $3: when crossed, the agent finishes in library-only mode.
  - `BUDGET_USD_DAILY`, default $20: when crossed, new analyses are refused with a friendly message. Examples stay viewable.
- **The model ID for each role comes from an environment variable.**
  - During plumbing work, point `PHYSICIST_MODEL` at `muse-spark-1.3` ($1.25/$4.25 per 1M tokens, docs) to debug cheaply.
  - Wonder and Spark share the same endpoints, so it's a one-line switch. Always switch back to Wonder for real runs and the benchmark.

**Done when:** a full analysis replays offline with byte-identical results, and a test proves the daily cap blocks a new analysis.

---

# F. Frontend

## Part 24: Pages, components and UX

**What it does:** the app people actually see: phone-first, fast, and built to make the measurement and the reasoning legible.

**How to build it:**

- **Stack:**
  - Next.js (App Router), TypeScript, Tailwind, shadcn/ui.
  - TanStack Query, polling every 1 s while a job runs.
  - KaTeX for math.
  - uPlot for time series: thousands of points, synced cursors, error bands.
  - `@meta-sam/parser`, `@meta-sam/graphics`, `@meta-sam/video` and `@meta-sam/react` for segmented playback (docs). React 19 fits their React ≥18.2 <20 peer range.
- **Showing SAM output in the browser:**
  - Fetch the stored `sam/{object}.events.jsonl`.
  - Replay it as an async iterable into `parseVideoStream`.
  - Pass the result to the `Video` component, which handles frame-accurate playback with the mask overlay.
  - This puts Meta's own SAM tooling on screen.
- **Pages:**
  - `/` **Landing:**
    - a muted, looping ghost-overlay video;
    - the one-liner;
    - **Try an example** and **Run your own** buttons;
    - a three-step "how it works";
    - a link to the benchmark.
  - `/examples`: cards for the seeded experiments (thumbnail, system, headline result, confidence).
  - `/new`: the wizard from Part 4.
  - `/e/[id]`: the experiment workspace.
    - **Header:** title, status, confidence badge, review badge.
    - **Pipeline stepper** while running, with phase names, elapsed time, and "you can leave this page".
    - Tabs:
      - **Result:**
        - the law card (equation, quantities ± uncertainty with units, confidence and its reasons);
        - "The story": the three pinned notebook moments;
        - **Prove it**, **What if** and **Export to simulator** buttons (Part 19).
      - **Measurement:**
        - the segmented video with toggles for the mask and the trail;
        - the calibration overlay with its grid;
        - x, y, θ against t;
        - quality flags.
      - **Hypotheses:** one card per model, sorted by held-out error. Each card shows:
        - name, mechanism and equation;
        - a "Wonder's first guess" badge;
        - a status chip;
        - a held-out RMSE bar with CI whiskers;
        - the parameter table, fit and residual plots, and the rejection reason.
      - **Review:** a thread of reviewer issues and Wonder's responses, with the final verdict.
      - **Notebook:** Part 20.
      - **Prove it:**
        - upload trial 2;
        - the ghost video and the error chart;
        - the divergence callout and **Why did it drift?**;
        - the v1/v2 toggle.
      - **What if:** Part 18.
  - `/benchmark`:
    - the results table;
    - a measured-vs-truth plot with error bars;
    - the coverage number and the model-selection accuracy;
    - the failure cases, linked to their experiments.
  - `/about`:
    - the architecture diagram;
    - each Meta model and exactly what it does;
    - limitations;
    - the privacy note.
  - `/admin` (passcode): jobs, failures, cost per experiment, cache hit rate, and the daily budget.
- **UX details:**
  - Single column under 640 px; tap targets at least 44 px.
  - Dark theme with a cyan ghost accent.
  - Skeleton loaders, and no layout shift when video loads.
  - Videos use `playsInline` and `preload="metadata"`.
  - Errors say what happened and what to do, with a **Retry** button.
  - Color contrast meets WCAG AA.
  - Lazy-load the simulator and the video libraries.
  - The passcode lives in `localStorage`, read and written inside try/catch, with a prompt if it's missing.

**Done when:** on a mid-range Android phone and an iPhone, a judge can go from the landing page to a finished example's ghost video in three taps, with no horizontal scrolling.

---

# G. Quality

## Part 25: The benchmark

**What it does:** proves Ghostline measures real physics correctly, and that its uncertainty is honest. This is the strongest answer to "does it work?"

**How to build it:**

- **The plan:** about 25 clips.
  - Free fall: 4 dense, 2 foam.
  - Pendulum: 3 small-amplitude, 2 large-amplitude.
  - Ramp: 3 block, 2 ball.
  - Bounce: 4.
  - Spin-down: 2, for the story only.
  - Failure cases: 3, deliberately bad.
- **Ground-truth methods:**

| Quantity | Truth |
|---|---|
| g (free fall, pendulum) | Local gravity from latitude φ: g = 9.780327 × (1 + 0.0053024 sin²φ − 0.0000058 sin²2φ) m/s², minus 3.086×10⁻⁶ per meter of altitude |
| Pendulum sub-checks | L by tape (±1 mm); T by stopwatch over 20 periods |
| μ_k (ramp, block) | Tilt the board until the block slides at constant speed: μ_k = tan θ. Five repeats, mean ± SD. |
| Rolling ball | Theory: a = (5/7) g sin α, so β = 2/5 |
| e (bounce) | Audio: e = Δt_{n+1}/Δt_n between impact clicks, from a normal-speed recording, averaged over five drops |
| Failure cases | Expected result is low confidence, for the right reason |

- **Files:**
  - `bench/manifest.csv`: clip_id, system, object, setup notes, truth, truth_sigma, method, file.
  - `bench/audio_bounce.py`: band-pass, energy-based onset detection, Δt series.
- **The runner, `bench/run.py --live|--replay`:**
  1. Runs every clip through the deployed pipeline.
  2. Collects the results.
  3. Writes `bench/results.csv` and `bench/results.md`, the table that goes into the README and the pitch.
  4. Uploads the benchmark rows.
- **Metrics:**
  - error, and z = (measured − truth)/σ_combined;
  - **coverage:** the share of clips whose 95% interval contains the truth;
  - **model-selection accuracy:** the right physics chosen (FF0 for the dense ball, a drag model for the foam ball, R1 for the block, R2 for the ball);
  - the failure cases flagged correctly.
- **Cost:** about $1.50 per clip including Wonder, so roughly $40 for one full live pass. Every later pass replays for free unless code changes the inputs.

**Done when:** `bench/results.md` exists with real numbers, and the pitch's `[bracket]` placeholders are filled from it.

## Part 26: Testing

**What it does:** catches regressions without spending money, and catches real-world failures before judges do.

**How to build it:**

- **Unit tests (pytest):**
  - the timing classifier, on synthetic timestamp arrays: CFR, jitter, mixed and re-timed;
  - the homography and offset correction;
  - mask features, on synthetic discs and rectangles;
  - angle unwrapping and the aliasing flag;
  - each library model, closed form against ODE;
  - parameter recovery;
  - selection on synthetic data;
  - interval coverage, over 100 synthetic runs;
  - the harness contract, run locally with `subprocess` so tests don't need Modal;
  - the AST code checker;
  - the SAM parser glue, on the fixture saved in Part 1;
  - the cost meter;
  - the budget guard;
  - the simulator export's unit conversions and the Bullet damping formula.
- **Synthetic video generator (`synth/render.py`):**
  - Draws a textured background and an A4 quad warped by a chosen camera pose.
  - Moves a shaded object according to a chosen model, with motion blur (averaged sub-frames) and noise.
  - Encodes H.264 at a chosen fps, optionally with a rotation tag.
  - Used for end-to-end tests: a few with live SAM, the rest in replay.
- **Integration:** the whole pipeline in replay mode, in CI.
- **Frontend:** `vitest` for simulator parity and the API client, plus a Playwright smoke test that loads an example and plays the ghost video.
- **Prompt evals (`evals/`):**
  - Run on 10 real clips, on demand, because they cost money.
  - Checks: the scene analysis is valid, the SAM prompt works, at least three competing models are proposed, rejections cite numbers, the conclusion matches the expected physics, and the reviewer catches the planted errors.
  - Results are stored with the prompt version.
- **Manual device checklist:**
  - iPhone Safari: HEVC, HDR, slow-mo, portrait.
  - Android Chrome.
  - Desktop Chrome and Safari.
  - A throttled network.
  - An upload resumed after a network drop.
  - Leaving and returning mid-job.
  - Two analyses running at once.

**Done when:** CI is green, and the device checklist passes on Nov 3.

---

# H. Shipping

## Part 27: Deployment and operations

**What it does:** gets everything running in production and keeps it alive through judging (Nov 4–8).

**How to build it:**

- **Modal (`backend/modal_app.py`):**
  - The base image is `debian_slim(python_version="3.12")` with ffmpeg from apt and the Python dependencies. Debian's ffmpeg includes the zimg library that the tone-map filter needs. **Verify** with `ffmpeg -filters | grep zscale`.
  - The fit image is the scientific stack only.
  - Plus the Volume `ghostline-data` and the secret `ghostline`.
  - Deploy with `modal deploy backend/modal_app.py`, once to `dev` and once to `main`.
- **Vercel:**
  - `NEXT_PUBLIC_API_BASE` holds the Modal API URL.
  - Production branch `main`; previews for other branches.
  - FastAPI's CORS allows the production and preview origins.
- **Supabase:** run migrations from CI or by hand, and use the pooled connection string.
- **Keep warm during judging:**
  - Set `min_containers=1` on `api` from Nov 4 at 9 PM ET until Nov 9, so the first request isn't a cold start.
  - At 0.5 CPU and 512 MiB, that costs about $0.03 per hour, or about $4 for the window.
  - Workers start in seconds, which is fine behind the progress stepper.
- **Observability:**
  - Structured JSON logs carrying experiment_id, job_id and phase.
  - Modal's dashboard.
  - The `/admin` page.
- **Backups:**
  - Every night, a scheduled Modal Function copies the examples and benchmark experiments to `/backups` on the Volume, and you `modal volume get` them to your laptop.
  - Supabase daily backups (free tier) plus a manual `pg_dump` before submission.
- **Runbook (`docs/runbook.md`):**
  - **Meta API down:** set `LLM_MODE=replay`. Examples still work. Show a banner saying new analyses are paused.
  - **Reviewer endpoint down:** analyses finish as "Not reviewed".
  - **Budget cap hit:** raise it, or wait for tomorrow.
  - **Stuck job:** the heartbeat timeout marks it failed; press **Retry**.
- **Modal credits:** with SAM on the Meta API, the self-hosted SAM fallback in the credits estimate (about $16) isn't needed.

**Done when:** the production URL works from a phone on cellular, and a deliberately killed worker recovers through **Retry**.

## Part 28: Security, privacy and abuse

**What it does:** protects your keys and credits, the people who appear in videos, and the app itself from hostile input.

**How to build it:**

- **Secrets:**
  - Only in the Modal secret and in server-side Vercel environment variables.
  - Nothing secret ever goes in `NEXT_PUBLIC_*`.
- **Access:**
  - Reading the examples is public.
  - Creating, uploading and analyzing require the passcode, which judges get in the submission notes.
  - A limit of 10 analyses per IP per day.
  - The daily budget cap (Part 23).
- **Uploads:**
  - `ffprobe` must succeed.
  - Video streams only.
  - At most 1 GiB and 15,000 frames.
  - Filenames are never used as paths.
- **Sandbox:**
  - No network.
  - CPU, memory and time limits.
  - The import allowlist.
  - Output size caps.
- **Prompt injection:**
  - Text visible in a video (a sign saying "ignore your instructions") can't change any number, because the harness computes every number and the orchestrator validates every claim.
  - The reviewer sees only artifacts.
- **Privacy:**
  - Experiments are private by default, behind unguessable UUIDs.
  - Media routes check visibility.
  - Meta Files uploads expire after 7 days.
  - Uploads that aren't examples or benchmark clips are deleted after 14 days by a daily scheduled function.
  - A **Delete experiment** button.
  - The About page says clips are processed by Meta Model API (Standard mode, so not used to train Meta's models, docs) and stored on Modal.

**Done when:** no secret appears in the built JavaScript (grep the `.next` output), and requests without the passcode are refused.

## Part 29: Demo, video and submission

**What it does:** turns a working app into a winning submission. The demo is 25% of the score.

**How to build it:**

- **Seeded examples, ready by Nov 2:**
  1. **Spin-down:** Wonder's exponential-decay guess refuted on held-out frames. This is the hero story. Use whichever system produced a genuine refutation on film.
  2. **Pendulum:** the ghost on trial 2, plus Mars in What if.
  3. **Ramp:** μ_k for the block, and 5/7 for the ball.
  4. **Bounce:** e, checked against audio.
  5. **Free fall:** the dense ball gives constant g; the foam ball switches to a drag model.
  6. **Can't tell:** an out-of-plane pendulum flagged as low confidence.
- **Demo video:** about 2–3 minutes. Check the submission form for the exact limit.

| Time | Content |
|---|---|
| 0:00–0:10 | The ghost locked to a real pendulum bob. On-screen text: "This ghost is a prediction." |
| 0:10–0:25 | The problem: lab sensors cost hundreds; video analysis is tedious. |
| 0:25–0:50 | Film, upload, and the stepper running; SAM 3.1 masks appear; the A4 grid snaps into place. |
| 0:50–1:30 | The spinner: Wonder predicts exponential decay, the held-out frames disagree, the rejection with numbers, Spark Open's challenge and Wonder's answer. |
| 1:30–1:55 | Prove it: a new trial, the ghost tracks, a drift, the diagnosis, and v2. |
| 1:55–2:15 | What if: the same pendulum on Mars. Then one tap exports the ramp's measured friction to PyBullet, and the replay check shows the simulated block matching the real one. |
| 2:15–2:35 | The benchmark table, "our 95% intervals contained the truth in X of N clips", and the "can't tell" example. |
| 2:35–2:45 | Name, link, and the Meta models used. |

- **Recording:** screen-record on the phone (iOS screen recording) for the capture flow, and on desktop for the detailed views. Voiceover, burned-in captions, quiet music.
- **Live finals:**
  - The examples open in tabs.
  - One live run, using a pre-filmed clip already on the phone.
  - A replay-mode switch if the venue network fails.
- **Submission checklist:**
  - A public GitHub repo whose README covers setup, the architecture diagram, how each Meta model is used, the benchmark table, limitations and the license.
  - The deployed URL and the passcode.
  - The demo video (YouTube unlisted or Loom).
  - The pitch from `ghostline-pitch.md`, with every bracket filled and the notes section deleted.
  - Track: Indie. Theme: Push the Frontier.
  - Screenshots.

**Done when:** everything is submitted by Nov 4, 5 PM ET, four hours before the deadline.

## Part 30: Build order and cut line

**What it does:** sequences the work so a complete, demoable product exists early, and decides in advance what to drop if time runs short.

**How to build it:**

| Day | Build | End-of-day checkpoint |
|---|---|---|
| Tue Oct 27 (evening) | Part 1 smoke test, Part 2 scaffold. Film three test clips (Part 3). | All eight smoke checks answered |
| Wed Oct 28 | Parts 5, 7, 8 and 9. Upload spike (Part 4). | First g from a drop, in meters, plotted |
| Thu Oct 29 | Parts 10, 11 and 12, with synthetic tests | Library selection passes the synthetic suite |
| Fri Oct 30 | Parts 6, 13, 20, 21 and 22 | One clip goes end to end through the API with a notebook |
| Sat Oct 31 | Part 14, Part 24 core (wizard, workspace), first production deploy (Part 27) | A full analysis works on the production URL from a phone |
| Sun Nov 1 | Part 16 (ghost). **Benchmark filming session** in daylight (Part 25). | The first ghost video renders |
| Mon Nov 2 | Parts 15, 17, 18 and 19; benchmark runs; seed examples (Part 29) | `bench/results.md` filled in; six examples live |
| Tue Nov 3 | Polish, the landing, about and benchmark pages, Part 28, device QA (Part 26). **Feature freeze at 9 PM.** | Device checklist passes |
| Wed Nov 4 | Demo video, README, pitch numbers. Submit by 5 PM ET. | Submitted |

**Cut line.** If you're behind, cut from the top of this list first:

1. The interactive ghost overlay in the browser. Keep the server-rendered MP4.
2. The PyBullet replay check. Keep the export files.
3. Editing model code from the notebook. Keep re-running with a different split.
4. Wonder-written custom models. Go library-only, with Wonder still choosing, ranking and rejecting.
5. The bounce system.
6. Ghost v2 from the revision loop. Keep the diagnosis text.
7. What if for anything other than the pendulum.
8. Export to simulator. If you cut it, rewrite the robotics bullet in the pitch's "Beyond the classroom" section as future work.

**Never cut:**
- the measurement pipeline;
- held-out selection;
- a genuinely refuted hypothesis on film;
- the reviewer;
- the ghost video;
- the benchmark table;
- the seeded examples;
- the demo video.

## Part 31: Risks and open questions

| Risk | Impact | Mitigation | When you'll know |
|---|---|---|---|
| Wonder's video input uses many tokens | Cost | Send sampled frames instead of the video | Part 1, check 2 |
| Tool results can't carry images | Agent design | Attach images in a follow-up user message | Part 1, check 3 |
| Spark Open endpoint lacks JSON-schema output | Reviewer parsing | Prompted JSON with a repair retry | Part 1, check 6 |
| iOS re-times or transcodes slow-mo uploads | Wrong time base, so wrong g | Default to 60 fps, the timing model (Part 5), and a physics sanity check | Part 1, check 8 |
| The SAM mask origin isn't what the docs' examples suggest | Misaligned centroids | Overlay check on the first clip | Part 1, check 5 |
| The object moving in front of the sheet biases scale | Error in g | Stand at least 2 m back, use 2x zoom, offset correction (Part 8) | First drop clip |
| Wonder guesses right the first time on the spinner | No refutation story | Use whichever system genuinely refutes; never script it | Benchmark day |
| Meta API spend runs over budget | Money | Spark 1.3 for plumbing work, replay cache, budget caps | Part 23 meter |
| Judges hit a cold or broken back end | Demo score | Keep warm, seeded examples, replay mode, runbook | Nov 3 QA |
| PyBullet won't install on the fit image's Python version | No replay check | Pin the fit image to a Python version PyBullet ships wheels for, or run the check in MuJoCo (`pip install mujoco`) | First export build (Part 19) |

---

# Appendices

## A. Environment variables

**Modal secret `ghostline`:**

| Variable | Value |
|---|---|
| `MODEL_API_KEY` | Meta Model API key |
| `META_BASE_URL` | `https://api.meta.ai/v1` |
| `PHYSICIST_MODEL` | `muse-wonder-1.0` |
| `SCENE_MODEL` | `muse-wonder-1.0` |
| `SAM_MODEL` | `sam-3.1` |
| `REVIEWER_BASE_URL` | Modal Shared Endpoint URL |
| `REVIEWER_MODEL` | Name from the dashboard or `/v1/models` |
| `REVIEWER_API_KEY` | `wk-<id>.ws-<secret>` |
| `DATABASE_URL` | Supabase pooled connection string |
| `APP_PASSCODE` | Passcode for write routes |
| `BUDGET_USD_DAILY` | `20` |
| `BUDGET_USD_PER_EXPERIMENT` | `3` |
| `LLM_MODE` | `live`, `record` or `replay` |
| `CORS_ORIGINS` | Vercel production and preview origins |

**Vercel:** `NEXT_PUBLIC_API_BASE`.

## B. Prompt drafts (versioned in `backend/ghostline/prompts/`)

**Scene analyst (Wonder)**
> You are the lab assistant for Ghostline, a phone-based physics lab. You will watch a short video of a physics experiment and decide what to measure. The clip's real duration is {duration}s and it was captured at {capture}. The user says: "{hint}".
> Identify the moving object(s), any marker or pivot, and the sheet of paper used for scale. For each object to track, write a SAM prompt: a short noun phrase naming one concrete object, with a distinguishing color or material ("orange foam ball", "blue dot sticker"). No verbs, questions, lists, negation or position words.
> Before you see any measurements, list 2–4 physical mechanisms that could govern this motion and, for each, the observable signature that would confirm or refute it.
> Report problems honestly: camera movement, motion toward or away from the camera, occlusion, the object leaving the frame, blur, a sheet that isn't fully visible. If the clip can't be measured, set proceed to false and say why.
> Return JSON matching the SceneAnalysis schema.

**Physicist (Wonder)**
> You are the physicist in Ghostline. Your job is to find the physical law that governs a measured motion, using the scientific method in the open, because students will read your notebook.
> Rules:
> 1. Before any fit, propose 3–5 competing models with different physical mechanisms. Name your favorite and, for every model, the result that would refute it.
> 2. You only see the training window. A harness you can't modify scores each model on held-out data you never see. The best held-out prediction wins, not the best fit.
> 3. Every rejection must cite a held-out error or a residual pattern from the harness. Never state a number the harness didn't produce.
> 4. Don't add a parameter without a physical mechanism for it.
> 5. If models are statistically tied, say the data can't distinguish them and suggest what experiment would.
> 6. Keep each narrative to 2–4 short sentences a high-school student can follow. Use LaTeX for equations.
> Use the tools. Finish with submit_conclusion.

**Reviewer (Spark Open)**
> You are a skeptical referee reviewing a physics claim before it's published to students. You see the claim, the comparison of all candidate models on held-out data, uncertainty budgets, data-quality flags, and two plots (fit and residuals). You don't see the author's reasoning.
> Check: (1) the winner's held-out advantage versus its uncertainty; (2) structure in the residuals; (3) physical plausibility and parameters at bounds; (4) whether the intervals are believable given calibration; (5) quality flags that undercut the claim; (6) an overstated narrative; (7) a simpler untested explanation.
> Be specific and cite numbers. Don't invent problems: if the claim holds, accept it. Return JSON matching the Review schema.

**Divergence diagnosis (Wonder)**
> The ghost (your model's prediction) drifted from the real object on a new trial. You see the error over time and frames around the moment it diverged, with the ghost and the real object both visible. Diagnose the most likely missing or wrong physics in one or two sentences, propose a revised model, and state what its prediction should look like if you're right.

## C. Core schemas (Pydantic, abbreviated)

```python
class TrackedObject(BaseModel):
    key: str; sam_prompt: str
    role: Literal["moving_body", "marker", "pivot", "surface"]; why: str

class SceneAnalysis(BaseModel):
    system_type: Literal["free_fall", "projectile", "pendulum", "ramp", "bounce", "spin_down", "oscillator", "other"]
    description: str
    tracked_objects: list[TrackedObject]
    calibration: dict            # sam_prompt, found, fully_visible, in_motion_plane, note
    object_properties: dict      # shape, material_guess, approx_size_cm
    initial_hypotheses: list[dict]   # name, mechanism, observable_signature
    quality_issues: list[dict]       # issue, severity, note
    proceed: bool; reason_if_not: str | None

class ModelProposal(BaseModel):
    id: str; name: str; mechanism: str; equations_latex: str
    library_ref: str | None; code: str | None; params: dict[str, ParamSpec] | None
    prior_rank: int; refuting_signature: str

class Quantity(BaseModel):
    name: str; value: float; ci_low: float; ci_high: float; unit: str

class Conclusion(BaseModel):
    winner_id: str; quantities: list[Quantity]
    rejected: list[dict]         # id, reason, evidence (must reference harness numbers)
    confidence: Literal["high", "medium", "low"]; caveats: list[str]; narrative_md: str

class ReviewIssue(BaseModel):
    id: str; severity: Literal["minor", "major"]; claim: str; evidence: str
    requested_action: Literal["run_model", "refit", "lower_confidence", "add_caveat", "none"]; detail: str

class Review(BaseModel):
    verdict: Literal["accept", "accept_with_caveats", "challenge"]
    issues: list[ReviewIssue]; summary: str

class ExportedParam(BaseModel):
    name: str; value: float; ci95: tuple[float, float]; unit: str
    definition: str; from_model: str

class SimExport(BaseModel):
    schema_version: Literal["ghostline.sim/1"]
    experiment_url: str; object: str; surface: str | None
    confidence: Literal["high", "medium", "low"]; mass_kg: float | None
    parameters: list[ExportedParam]
    randomization: dict[str, tuple[float, float]]   # from the 95% intervals
    valid_for: dict                                 # measured speed, angle and amplitude ranges
    replay_check: dict | None                       # simulator, quantity, simulated, measured, passed
```

## D. Reference docs used

- Meta Model API
  - Media segmentation: https://dev.meta.ai/docs/media-segmentation
  - SAM prompts, limits and errors: https://dev.meta.ai/docs/sam/segmenting
  - SAM output grammar: https://dev.meta.ai/docs/sam/reading-segmentation
  - SAM client libraries: https://dev.meta.ai/docs/sam/client-libraries
  - Video understanding: https://dev.meta.ai/docs/video-understanding
  - File handling: https://dev.meta.ai/docs/file-handling
  - Pricing and rate limits: https://dev.meta.ai/docs/pricing-rate-limits
  - Structured output: https://dev.meta.ai/docs/structured-output
  - Tool calling: https://dev.meta.ai/docs/tool-calling
  - Reasoning: https://dev.meta.ai/docs/reasoning
  - Prompt caching: https://dev.meta.ai/docs/prompt-caching
  - Error handling: https://dev.meta.ai/docs/error-handling
- Modal
  - Web functions (4 GiB request bodies, proxy auth): https://modal.com/docs/guide/webhooks
  - Request timeouts (150 s): https://modal.com/docs/guide/webhook-timeouts
  - Sandboxes: https://modal.com/docs/guide/sandboxes
  - Sandbox networking (`block_network`): https://modal.com/docs/guide/sandbox-networking
  - Sandbox file access: https://modal.com/docs/guide/sandbox-files
  - Endpoints and pricing: https://modal.com/docs and https://modal.com/pricing
