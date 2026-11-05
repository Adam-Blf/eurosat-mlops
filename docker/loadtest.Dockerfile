# Load generator that runs inside the compose network, so measurements do not include the
# host-to-container port forwarding of Docker Desktop. It only ever targets the local gateway.
FROM python:3.12.8-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 HOME=/tmp
RUN pip install "locust==2.47.0" && useradd --system --uid 10001 --no-create-home load
WORKDIR /load
COPY loadtest/guard.py loadtest/locustfile.py ./
COPY loadtest/samples ./samples
USER 10001
ENTRYPOINT ["locust", "-f", "locustfile.py", "--headless", "--host", "http://gateway:8080/api"]
