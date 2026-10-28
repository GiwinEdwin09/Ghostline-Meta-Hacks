"""Modal app: image, storage, secrets and functions (build plan Parts 21 and 27).

Run every Modal command from the repo root, because uv_sync reads pyproject.toml and uv.lock
from the current directory:

    uv run modal serve backend/modal_app.py    # live-reloading dev URL
    uv run modal deploy backend/modal_app.py   # production
"""

import modal

app = modal.App("ghostline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_sync(extra_options="--no-dev")
    # The default ignore drops non-Python files, which would leave out the prompt templates.
    .add_local_python_source("ghostline", ignore=["**/*.pyc"])
)

data = modal.Volume.from_name("ghostline-data", create_if_missing=True)
secrets = [modal.Secret.from_name("ghostline")]


@app.function(
    image=image,
    volumes={"/data": data},
    secrets=secrets,
    cpu=0.5,
    memory=512,
    max_containers=5,
)
@modal.asgi_app()
def api():
    from ghostline.api.app import create_app

    return create_app()
