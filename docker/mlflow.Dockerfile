FROM python:3.12.8-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 HOME=/tmp
RUN pip install "mlflow==3.11.1" "psycopg2-binary==2.9.9" \
    && useradd --system --uid 10001 --no-create-home mlflow \
    && mkdir /mlartifacts && chown 10001 /mlartifacts
USER 10001
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=5 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/health', timeout=2)"]
