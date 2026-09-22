FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheelhouse .


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TFM_STATE_DIR=/var/lib/tfm_multiagent

WORKDIR /app

COPY --from=builder /wheelhouse /wheelhouse

RUN python -m pip install --no-cache-dir /wheelhouse/*.whl \
    && rm -rf /wheelhouse \
    && groupadd --system tfm \
    && useradd --system --gid tfm --home-dir /nonexistent --no-create-home tfm \
    && mkdir -p "${TFM_STATE_DIR}" \
    && chown -R tfm:tfm "${TFM_STATE_DIR}"

USER tfm

EXPOSE 8000
VOLUME ["/var/lib/tfm_multiagent"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
