FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md config.yaml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 SHELF_BENCH_CONFIG=/app/config.yaml
ENTRYPOINT ["shelf-bench"]
