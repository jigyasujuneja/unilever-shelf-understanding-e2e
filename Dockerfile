FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md config.yaml gt-mapping.yml ./
COPY src ./src
COPY experiments ./experiments
COPY scripts ./scripts
# Optional pip extras, e.g. --build-arg EXTRAS=cv (TensorFlow for the legacy CV prediction server)
ARG EXTRAS=
RUN mkdir -p /app/results && pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"
ENV PYTHONUNBUFFERED=1 SHELF_BENCH_CONFIG=/app/config.yaml
ENTRYPOINT ["shelf-bench"]
