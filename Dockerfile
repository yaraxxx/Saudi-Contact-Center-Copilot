FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY cc_copilot ./cc_copilot
COPY data ./data
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" ENABLE_TRACING=0
EXPOSE 8000
CMD ["uvicorn", "cc_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
