FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir /app
WORKDIR /app

# Outside /app, so copying the project below cannot replace it with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# The dependencies alone first, so a change to the code does not install them again.
COPY pyproject.toml uv.lock /app/
# torch for an NVIDIA GPU: PyPI's own wheels are the CUDA 13 build and bring the CUDA
# runtime, so the image needs no CUDA of its own, only a host driver (>= 580) and `--gpus all`.
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/opt/venv/bin:$PATH"

# The nnInteractive weights, baked in before the code, so a pod needs no network and
# a change to the code does not download them again. They are CC BY-NC-SA 4.0 (DKFZ):
# their LICENSE comes with them, and the README says what that allows.
ENV NNINTERACTIVE_MODEL_DIR=/opt/nninteractive
RUN python -c "from nnInteractive.model_management import ensure_model_available; ensure_model_available('nnInteractive_v1.0')" \
    && test -f /opt/nninteractive/models/nnInteractive_v1.0/fold_0/checkpoint_final.pth \
    && test -f /opt/nninteractive/models/nnInteractive_v1.0/LICENSE \
    && rm -rf /opt/nninteractive/downloads /root/.cache/huggingface \
    && chmod -R a+rX /opt/nninteractive
# Nothing at run time asks Hugging Face: the app reads the weights from the folder.
ENV HF_HUB_OFFLINE=1

# The whole project: the app target may be any module in it, not only app.py.
COPY . /app
# And the project itself, so `path_nninteractive` (in src/) is importable.
RUN uv sync --frozen --no-dev

# The version of this build, when it is not the one the source declares: a build from
# a branch is passed it by `arkitekt plugin release`. Empty for a release proper.
ARG ARKITEKT_APP_VERSION=""
ENV ARKITEKT_APP_VERSION=$ARKITEKT_APP_VERSION
