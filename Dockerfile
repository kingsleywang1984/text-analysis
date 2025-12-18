#
# Multi-stage build:
# - builder installs dependencies with uv into an isolated venv
# - runtime copies only the venv + app code (keeps build tooling out of final image)
#

FROM public.ecr.aws/lambda/python:3.12 AS builder

WORKDIR ${LAMBDA_TASK_ROOT}

# Install uv (build-time only)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir uv

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime deps only, frozen lockfile
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PROJECT_ENVIRONMENT=${LAMBDA_TASK_ROOT}/.venv
RUN uv sync --frozen --no-dev && \
    rm -rf "${UV_CACHE_DIR}" /root/.cache/uv /root/.cache/pip


FROM public.ecr.aws/lambda/python:3.12 AS runtime

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy runtime venv from builder
COPY --from=builder ${LAMBDA_TASK_ROOT}/.venv ${LAMBDA_TASK_ROOT}/.venv

# Copy source
COPY src ./src

# Ensure imports resolve and venv is active
# Expose both source code and venv site-packages on PYTHONPATH so the default Lambda
# runtime interpreter can import installed dependencies without needing to activate the venv.
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}:${LAMBDA_TASK_ROOT}/src:${LAMBDA_TASK_ROOT}/.venv/lib/python3.12/site-packages:${LAMBDA_TASK_ROOT}/.venv/lib64/python3.12/site-packages
ENV VIRTUAL_ENV=${LAMBDA_TASK_ROOT}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# Lambda handler entrypoint
CMD ["src.handler.lambda_handler"]
