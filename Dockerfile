FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md config.yaml gt-mapping.yml ./
COPY src ./src
COPY experiments ./experiments
COPY scripts ./scripts
COPY docs ./docs
COPY tests ./tests
COPY data ./data
# Optional pip extras, e.g. --build-arg EXTRAS=cv (TensorFlow for the legacy CV prediction server)
ARG EXTRAS=
RUN mkdir -p /app/results && pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"
COPY results/locust_load_test_report.json ./results/locust_load_test_report.json
ENV PYTHONUNBUFFERED=1 SHELF_BENCH_CONFIG=/app/config.yaml
ENTRYPOINT ["shelf-bench"]
