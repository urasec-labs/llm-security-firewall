FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir "uv==0.9.24"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY artifacts/api-server/firewall ./firewall

RUN useradd --create-home --uid 10001 appuser
USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/healthz', timeout=2)"

CMD ["/app/.venv/bin/uvicorn", "--app-dir", "/app", "firewall.main:app", "--host", "0.0.0.0", "--port", "8080"]