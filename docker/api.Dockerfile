# syntax=docker/dockerfile:1.7
FROM python:3.12.8-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install .

FROM python:3.12.8-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 HOME=/tmp GIT_PYTHON_REFRESH=quiet
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY models/release ./models/release
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]
CMD ["uvicorn", "eurosat_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
