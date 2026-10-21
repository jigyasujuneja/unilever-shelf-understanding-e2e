# Google MaxViT-Tiny Deployment & Observability Guide

## 1. Overview & Architecture

This guide details the production deployment, telemetry integration, and evaluation of **Google MaxViT Tiny (`maxvit_tiny_tf_224`)** for fine-grained 1,922-class Brand__Variant classification within the `marketshare_poc` shelf intelligence system.

```
                                  Input Shelf Crop (224x224 RGB)
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │      MaxViT Multi-Axis Attention Backbone    │
                         │   - Block Attention (Local Window Self-Attn) │
                         │   - Grid Attention (Global Dilated Self-Attn)│
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Linear Head (1,922 Canonical Classes)      │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │         Top-5 Margin Decision Filter         │
                         │   - Top-1 Confidence & Top-2 Margin Gap      │
                         │   - OpenTelemetry Trace Span Generation      │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                                   Predicted SKU + Canonical Meta
                                   (Brand, Base Variant, Category)
```

---

## 2. Model Artifacts & GCS Storage

All model artifacts are stored centrally in Google Cloud Storage:

- **Storage Bucket / Path:** `gs://marketshare-datasets/models/maxvit_variants/`
- **Native PyTorch State Dict:** `best_maxvit_brand_variant.pth` (118.8 MB)
- **TorchScript Traced Model:** `traced_maxvit_brand_variant.pt` (122.9 MB)
- **Class & Taxonomy Index:** `variant_classes.json` (1,922 classes with canonical Unilever SKU metadata)

---

## 3. Vertex AI Model Registry Registration

The MaxViT model is registered in Google Cloud Vertex AI Model Registry with custom metadata labels and the unified container specification.

### Via Automation Script
```bash
python3 scripts/deploy_maxvit_endpoint.py --project ul-fs-s-sandbx-16-prj --region us-central1
```

### Via Google Cloud CLI (`gcloud`)
```bash
gcloud ai models upload \
  --project=ul-fs-s-sandbx-16-prj \
  --region=us-central1 \
  --display-name=marketshare-maxvit-tiny-tf-224 \
  --artifact-uri=gs://marketshare-datasets/models/maxvit_variants \
  --container-image-uri=us-central1-docker.pkg.dev/ul-fs-s-sandbx-16-prj/cloud-run-source-deploy/shelf-bench:latest \
  --container-args="serve" \
  --container-ports=8080 \
  --container-health-route="/" \
  --container-predict-route="/api/predict" \
  --description="[market_share/classification] Google MaxViT Tiny 1,922-Class Variant Classifier."
```

### Registered Model Resource
- **Model Resource ID:** `projects/663686437672/locations/us-central1/models/187489272524177408`
- **Model Display Name:** `marketshare-maxvit-tiny-tf-224`
- **Console Link:** [Vertex AI Model 187489272524177408](https://console.cloud.google.com/vertex-ai/models/locations/us-central1/models/187489272524177408?project=ul-fs-s-sandbx-16-prj)

---

## 4. Vertex AI Endpoint Deployment

To serve low-latency online predictions for shelf crops, an endpoint is created with dedicated `n1-standard-4` compute resources.

### Create Endpoint
```bash
gcloud ai endpoints create \
  --project=ul-fs-s-sandbx-16-prj \
  --region=us-central1 \
  --display-name=marketshare-maxvit-vertex-endpoint
```

### Deploy Model to Endpoint
```bash
gcloud ai endpoints deploy-model 3648281285786337280 \
  --project=ul-fs-s-sandbx-16-prj \
  --region=us-central1 \
  --model=187489272524177408 \
  --display-name=marketshare-maxvit-tiny-tf-224 \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100
```

### Deployed Endpoint Resource
- **Endpoint Resource ID:** `projects/663686437672/locations/us-central1/endpoints/3648281285786337280`
- **Endpoint Display Name:** `marketshare-maxvit-vertex-endpoint`
- **Console Link:** [Vertex AI Endpoint 3648281285786337280](https://console.cloud.google.com/vertex-ai/online-prediction/endpoints/3648281285786337280?project=ul-fs-s-sandbx-16-prj)

---

## 5. Production Telemetry & Observability

MaxViT is fully instrumented with OpenTelemetry (`ctx.trace.step`), capturing rich inference diagnostics for every crop:

1. **Top-5 Candidate Shortlists:** Softmax probability distribution over the highest-scoring candidate SKUs.
2. **Top-2 Decision Margin:** Mathematical difference between top-1 and top-2 candidate probabilities:
   $$\text{Margin} = P(\text{Top-1}) - P(\text{Top-2})$$
   A margin $> 0.35$ indicates high discriminative separation; a margin $< 0.10$ triggers potential fallback to Stage 2 multivariant resolution.
3. **Structured Cloud Tracing:** Spans are pushed directly to Google Cloud Trace:
   `https://console.cloud.google.com/traces/explorer;traceId=<TRACE_ID>?project=ul-fs-s-sandbx-16-prj`

---

## 6. Verification & Evaluation on `hul_golden`

To reproduce the benchmark on 1,000 images in the `hul_golden` test split:

```bash
python3 -m cli run \
  -a maxvit_classify \
  -m maxvit_tiny_tf_224 \
  -d hul_golden \
  --split test \
  --limit 1000 \
  --workers 4
```

### Benchmark Summary Metrics (`results/1021-121329-maxvit_classify-maxvit_tiny_tf_224-hul_golden`)

| Evaluation Dimension | Metric Value | Notes |
| :--- | :---: | :--- |
| **Exact SKU Accuracy (L5)** | **87.1%** | 871 / 1,000 correct predictions |
| **95% Confidence Interval** | `[84.9%, 89.0%]` | $N = 1,000$ random test split |
| **HUL Attribution (L1)** | **99.0%** | Only 1.2% false attribution rate |
| **Category Accuracy (L2)** | **97.3%** | Oral Care, Foods, Laundry, Skin, Hair |
| **Brand Accuracy (L3)** | **96.8%** | 98.4% HUL Brand / 95.2% Competitor Brand |
| **Packaging Type Accuracy** | **95.4%** | Tube, Bottle, Pouch, Carton, Jar, Bar |
| **Share of Shelf (SoS) Facing Error** | **0.2%** | Ground Truth 50.0% vs Predicted 50.2% |
| **Share of Shelf (SoS) Area Error** | **0.11%** | Ground Truth 55.01% vs Predicted 55.12% |
| **VLM Cost per Image** | **$0.000 (₹0.000)** | Zero generative token overhead |
| **Cloud Trace** | [`fd09d5a1c73e881c3242bae3a9425e49`](https://console.cloud.google.com/traces/explorer;traceId=fd09d5a1c73e881c3242bae3a9425e49?project=ul-fs-s-sandbx-16-prj) | Verified end-to-end trace span |
