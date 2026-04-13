# Custom Triton image with GLiNER2 dependencies pre-installed.
# Extends the official Triton Python backend image.
#
# Build:
#   docker compose build triton
#
# Why not use the stock image directly:
#   The Python backend (model_repository/gliner_guard/1/model.py) imports
#   gliner2 and its transitive deps (gliner, transformers, torch, pydantic).
#   None of these are in the stock Triton image.

FROM nvcr.io/nvidia/tritonserver:25.01-py3

WORKDIR /workspace

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Install gliner first (heavy: torch, transformers) then gliner2 from local src.
# torch is already in the base image; reinstalling ensures version compatibility.
RUN pip install --no-cache-dir \
        gliner \
        pydantic>=2.0.0

# Copy gliner2 source and install as editable package
# (the source is mounted at build time via the build context)
COPY repo_gliner2/ /workspace/repo_gliner2/
RUN pip install --no-cache-dir -e /workspace/repo_gliner2/

# ── Verify imports ─────────────────────────────────────────────────────────────
RUN python -c "import gliner2; import gliner; print('gliner2 OK:', gliner2.__version__)"
