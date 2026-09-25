FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md config.yaml ./
COPY configs ./configs
COPY data/splits ./data/splits
COPY results ./results
COPY src ./src
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 SHELF_BENCH_CONFIG=/app/config.yaml PORT=8080
ENTRYPOINT ["shelf-bench"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
