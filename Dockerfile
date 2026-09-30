FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt README.md config.yaml ./
COPY configs ./configs
COPY data/splits ./data/splits
COPY data/models ./data/models
COPY data/labeled_retail_benchmarks ./data/labeled_retail_benchmarks
COPY results ./results
COPY src ./src
COPY web ./web
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1 SHELF_BENCH_CONFIG=/app/config.yaml PORT=8080
ENTRYPOINT ["shelf-bench"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
