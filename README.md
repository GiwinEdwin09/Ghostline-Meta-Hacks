# Ghostline

**A physics lab in your phone, with an AI lab partner that shows its work.**

Film anything that moves. Ghostline measures it, proposes the physical laws that could explain it, rejects the ones the data disagrees with, and proves the winner by overlaying a ghost on a new clip that predicts where the object goes next.

Built for the Meta AI Developer Hackathon (Indie track, Push the Frontier).

> **Status:** starter skeleton. The full design, component by component, is in [docs/build-plan.md](docs/build-plan.md).

## How it works

1. **Watch.** Muse Wonder watches the clip and decides what to measure.
2. **Track.** SAM 3.1 segments the object in every frame; an A4 sheet in the frame turns pixels into meters.
3. **Hypothesize and test.** Wonder proposes competing physical models. A harness running in a Modal Sandbox fits each one and scores it on frames the model never saw.
4. **Review.** Muse Spark Open, on a Modal Shared Endpoint, challenges the claim before Ghostline reports it.
5. **Prove.** On a new trial, a ghost of the object follows the fitted law, and Ghostline measures how far reality drifts from it.

## Repository layout

```
backend/
  modal_app.py        Modal app: image, Volume, secret, functions
  ghostline/          Python package, one module per pipeline stage (see docs/build-plan.md, Part 2)
  migrations/         Postgres migrations
  tests/
web/                  Next.js app (src/app, src/components, src/lib)
scripts/              gen_api_types.sh and, later, the smoke test and seeding scripts
synth/  bench/  evals/   synthetic videos, benchmark, prompt evals (filled in later)
docs/build-plan.md    the build plan
```

## Local development

You need [uv](https://docs.astral.sh/uv/), Node 22 and ffmpeg.

**Back end** (from the repo root):

```bash
uv sync
uv run pytest
uv run uvicorn ghostline.api.app:create_app --factory --reload --port 8000
```

**Web app:**

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

The landing page shows whether it can reach the back end at `NEXT_PUBLIC_API_BASE`.

**API types:** after changing a back-end response model, run `npm run gen:api` in `web/` to regenerate `src/lib/api-types.ts` from the FastAPI schema.

## Configuration

Back-end settings are environment variables, listed with defaults in [.env.example](.env.example). In production they live in a Modal secret named `ghostline`; never commit real values.

## Deploying

**Back end on Modal** (run from the repo root, because the image installs from `pyproject.toml` and `uv.lock`):

```bash
uv run modal secret create ghostline MODEL_API_KEY=... CORS_ORIGINS=https://your-app.vercel.app
uv run modal deploy backend/modal_app.py
```

Use `uv run modal serve backend/modal_app.py` for a live-reloading development URL.

**Web app on Vercel:** set the project's root directory to `web` and `NEXT_PUBLIC_API_BASE` to the Modal API URL.

## CI

GitHub Actions runs ruff and pytest for the back end, and lint, tests and a production build for the web app, on every push.
