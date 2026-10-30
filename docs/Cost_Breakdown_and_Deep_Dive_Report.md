# **Cost Breakdown & Daytime Capacity Sizing**

## **Context**

Unilever India's retail vision platform is sized for a target scaling benchmark of **500,000 shelves/day (15.0M shelves/month)** across Modern Trade Market Share, General Trade Market Share, Display-Bay Merchandising, and Promotional Talkers—matching the 500,000 shelves/day scaling benchmark in *Existing Model Analysis* (`t.ipskbbbvu50h`).

In `Azure MarketShare Master Inventory.xlsx` (`Logic Apps` tab, rows 11–14), `marketsharestartcluster` starts the AKS inference cluster at `07:30 IST` daily and `marketsharestopcluster` stops it at `18:00 IST` Mon–Fri (`19:00 IST` Sat–Sun). All store visits and photo uploads happen during this **10.5-hour daytime store operating window (`07:30–18:00 IST`, averaging `13.2 shelves/sec` at `500,000 shelves/day`)**, while migrating from the current Azure footprint (`2 AKS clusters`, `18 hosted endpoints`, `4 NC16as_T4_v3 training VMs`, `17 ADF pipelines`, `20 Logic Apps`, `3 Azure SQL DBs`, `291.9 TiB Blob Storage`, and `Form Recognizer`) to Google Cloud (`PR01–PR18` in `GCP_MarketShare_Configuration.xlsx`).

This tab is dedicated exclusively to **Bottom-Up Cloud Billing SKU Pricing**, **Per-Shelf AI Inference Cost Receipts**, **12-Hour Daytime Provisioned Throughput (PT) GSU Sizing (`07:30–18:00 IST`)**, and **Shared Platform Overhead (`PR01–PR18` in LLD001 Sec 4.1.2)** at **500,000 shelves/day (15.0M/month)**. Path to Production, LLD001 `v0.1` $\rightarrow$ `v0.2` specification delta, Locust load testing, and the 4-stage cluster canary cutover are integrated into the *End-to-End Architecture Guide* (`t.erkk4n45aac9`).

### **Scope, Goals, and Pricing Disclaimer**

- **Goals:** Price every Google Cloud service bottom-up across AI inference (`PR06–PR08`, `PR14`) and shared platform components (`PR01–PR05`, `PR09–PR18`) at **500,000 shelves/day (15.0M/month)** (`1 USD = ₹92.54 INR`); and size stateless Cloud Run CPU workers and 12-hour shared Provisioned Throughput (PT) GSUs for the `07:30–18:00 IST` daytime store operating window.
- **Non-Goals:** Changing load balancers (`PR01–PR02`), Cloud Armor WAF, mobile apps, the Shared VPC perimeter, Cloud Composer schedules (`PR11`), Cloud SQL MySQL 8.4 table schemas (`PR15–PR17`), or dual-region CMEK Cloud Storage buckets (`PR18`)—all remain unchanged from LLD001 `v0.1`.
- **Enterprise Agreement (EA) Discount Schedule:** Discounted EA numbers apply Unilever's contracted Google Cloud product-family discount tiers alongside public Google Cloud SKU Catalog list prices (`https://cloud.google.com/skus`):
  - **Gen AI & Gen AI v2 (`gemini-3.5-flash-lite`, `gemini-embedding-2`, Provisioned Throughput GSUs):** **`20%`**
  - **Standard AI Platform (Vertex AI Custom Endpoints / L4 GPU Hosting):** **`23%`**
  - **Cloud SQL (Enterprise Compute, Enterprise Storage, Enterprise Plus Compute, Enterprise Plus Storage):** **`20%`**
  - **BigQuery, Cloud Composer (SKUs), and Pub/Sub:** **`25%`**
  - **Cloud Run & Cloud Storage Infrastructure:** **`35%`**

## **Key Findings**

- **Pure AI inference costs `$0.00160 / shelf` (`₹0.15 / shelf`) under EA Priority/PT (`$0.00132 / ₹0.12` per shelf on Standard PayGo 1.0x):** Applying the **`20%` Gen AI / Gen AI v2** and **`25%` Pub/Sub** discounts, grouping identical adjacent packs on a shelf row (`35` facings compressed into `~11.1` unique crop groups), and resolving `68%` of groups in memory with `gemini-embedding-2` holds pure AI inference at **500,000 shelves/day (15.0M/month)** to **`$24,000/month`** (`$19,800/month` Standard PayGo)—compared to **`$28,800/month`** (at `-23%` Standard AI Platform discount) for the `14× g2-standard-8` L4 GPU endpoints in Prod (`$38,600/month` including QA GPUs), which saturate above ~150K shelves/day and score only `42.32%` on `hul_golden` `test_5k`.
- **Adding unchanged `PR01–PR18` shared platform infrastructure (`$12,400/month` under the `20%` Cloud SQL and `25%` Composer/BigQuery/PubSub EA discounts) brings All-In project spend at 500,000 shelves/day to `$36,400/month` (`$0.00243 / ₹0.22` per shelf all-in):** The fixed platform components (`$6,500` Cloud SQL `PR15–PR17` at `-20%` off `$8,120` list, `$3,520` 300 TiB GCS `PR18`, `$1,100` App/Annotation Cloud Run `PR03–PR05`, and `$1,280` Load Balancers/Composer/Dataflow/BigQuery `PR01–PR02` & `PR09–PR12` with `-25%` on Composer & BigQuery) remain unchanged across all options, saving **`$4,800–$14,600/month`** against the existing GPU stack (`$41,200/month` Prod; `$51,000/month` Prod + QA) while doubling 5,000-crop SKU accuracy (`42.32%` $\rightarrow$ `82.30%–84.00%`).
- **Replacing the embedding API call with the distilled Local CPU ONNX Embedder lowers pure AI spend at 500,000 shelves/day to `$8,700/month` (`$0.00058 / ₹0.06` per shelf; `$21,100/month` all-in):** Running Stage 2A inside the Cloud Run CPU container eliminates `$7,500/month` in embedding API charges while preserving `82.1%–83.8%` 5,000-crop accuracy.
- **Eliminating `PR13` Document AI (`218,300` pages/mo in `GCP_MarketShare_Configuration.xlsx`; `2,183,900` pages/mo = `85k` pages/day in Azure PROD `6712-Marketshare-Form-Recognizer` for Toker OCR) saves `$2,600 – $3,275 / month` immediately:** Because Stage 2B (`gemini-3.5-flash-lite`) and our 3-class YOLO11s talker detector (`talker_price_tag_detect.py`) read promotional banners and rail tags natively (`90.02%` Talker F2), `PR13` is retired (`$0`).
- **Daytime store-window Provisioned Throughput (PT) requires `6–8 GSUs` (`07:30–18:00 IST`) with 50% overnight reuse:** Because store visits and uploads run during the 10.5-hour daytime store operating window (`07:30–18:00 IST` Mon–Fri; `07:30–19:00 IST` Sat–Sun), reserving `6 to 8 GSUs` (`21,300 – 28,400 tok/s` at `3,550 tok/s` per GSU) covers the `13.2 shelves/sec` daytime average (`~21,500 tok/s`), while the AI Foundations team can route overnight batch workloads (`18:00–07:30 IST`) onto those same GSUs.

*Figure 1: Panel A breaks down the per-shelf variable AI cost receipt ($0.00160 / ₹0.15 per shelf under EA Priority/PT at -20% Gen AI / Gen AI v2 and -25% Pub/Sub); Panel B compares monthly cloud spend at 500,000 shelves/day (15.0M/month) across pure AI inference and the unchanged $12,400/month PR01–PR18 shared infrastructure (-20% Cloud SQL, -25% Composer/BigQuery/PubSub).*

## **Cloud Billing SKUs & EA Discounts**

Every unit cost is calculated from the official [Google Cloud SKU Catalog](https://cloud.google.com/skus) (`cloudbilling.googleapis.com` for Vertex AI, Cloud Run, Cloud SQL, Cloud Composer, Pub/Sub, BigQuery, and Cloud Storage at `1 USD = ₹92.54 INR`).

### **1. Contracted Unilever EA Discount Schedule by Product Family**

| **Google Cloud Product Family** | **Covered Platform Components (`PR01–PR19`)** | **Contracted EA Discount** |
| --- | --- | :---: |
| **Cloud SQL Enterprise Compute & Storage** | `PR15–PR17` MySQL 8.4 Instances (`Processing`, `MT`, `GT`, `QA`) & Short-Lived PostgreSQL Vector Catalog DB | **`20%`** |
| **Cloud SQL Enterprise Plus Compute & Storage** | High-availability / read-replica Enterprise Plus database tiers | **`20%`** |
| **BigQuery, Cloud Composer (SKUs) & Pub/Sub** | `PR11` Cloud Composer orchestration, `PR19` Pub/Sub async ingestion queue, & BigQuery analytics | **`25%`** |
| **Standard AI Platform** | Legacy `14× g2-standard-8` L4 GPU custom model endpoints (`PR06–PR08` v0.1 baseline) | **`23%`** |
| **Gen AI & Gen AI v2** | `PR14` Vertex AI `gemini-embedding-2` & `gemini-3.5-flash-lite` (Standard `1.0x`, Priority `1.8x`, & PT GSUs) | **`20%`** |

### **2. Unit SKU Pricing (Retail List vs. Contracted Unilever EA Price)**

| **GCP Service & Role** | **Official Google Cloud SKU Description (cloud.google.com/skus)** | **Retail List Price (USD / INR)** | **Unilever EA Price (Contracted Discount)** |
| --- | --- | :---: | :---: |
| **Stage 1: SKU Detection (PR06–PR08)** | Cloud Run CPU (`257B-2A84-3396`) & Memory (`8A79-9F45-5F32`) in `europe-west3` | `$0.000018` (`₹0.0017`) / vCPU-s | `$0.000012` (`₹0.0011`) / vCPU-s (`-35%`) |
| **Stage 2A: gemini-embedding-2 (PR14 — Gen AI / v2)** | Vertex AI Multimodal Embedding Image (`330D-7FB0-87C5`) & Text (`EBBD-6991-B1BB`) | `$0.45` / 1M img tok (`₹41.64`) | `$0.36` / 1M img tok (`₹33.31`, **`-20%` Gen AI**) |
| **Stage 2B: gemini-3.5-flash-lite (Standard 1.0x — Gen AI v2)** | Global Input (`255D-4FF7-8DB3`, `$0.30/1M`), Output (`EF3E-6ED9-3CD1`, `$2.50/1M`), Cache (`$0.03/1M`) | `₹27.76` in / `₹231.35` out / `₹2.78` cache | `₹22.21` in / `₹185.08` out / `₹2.22` cache (**`-20%` Gen AI v2**) |
| **Stage 2B: gemini-3.5-flash-lite (Priority 1.8x — Gen AI v2)** | Priority Input (`A2BB-5D97-C5A2`, `$0.54/1M`), Output (`9331-9BC3-A888`, `$4.50/1M`), Cache (`A909-DA58-491D`, `$0.054/1M`) | `₹49.97` in / `₹416.43` out / `₹5.00` cache | `₹39.98` in / `₹333.14` out / `₹4.00` cache (**`-20%` Gen AI v2**) |
| **Stage 2B: gemini-3.5-flash-lite (12h Shared PT — Gen AI v2)** | Provisioned Throughput (PT) GSU (`3,550 tok/s` per GSU; 50% 12-hour daytime attribution) | `$2,000` / GSU / mo (`$1,000` for 12h) | `$1,600` / GSU / mo (`$800–$1,000` for 12h, **`-20%` Gen AI v2**) |
| **Shared DBs: Cloud SQL Enterprise / Plus (PR15–17)** | `3× MySQL 8.4` (`16 vCPU, 64–128 GB RAM, 1 TB` Compute + Storage) + Short-Lived Cloud SQL PostgreSQL Vector DB | `$8,120 / mo` | **`$6,500 / mo`** (`$6,496`, **`-20%` Cloud SQL**) |
| **Orchestration & Messaging: Composer, Pub/Sub & BQ (PR11, PR19)** | Cloud Composer (`PR11` SKUs), Pub/Sub (`PR19` ingestion), & BigQuery analytics | `$800 / mo` (Composer/BQ) + `$400 / mo` (Pub/Sub) | **`$600 / mo`** Composer/BQ + **`$300 / mo`** Pub/Sub (**`-25%`**) |
| **Legacy Baseline: Standard AI Platform GPUs (v0.1)** | `14× g2-standard-8` NVIDIA L4 GPU Endpoints (Prod `$37,400` list / Prod+QA `$50,100` list) | `$37,400 – $50,100 / mo` | **`$28,800 – $38,600 / mo`** (**`-23%` Std AI Platform**) |
| **Shared Storage: Cloud Storage 300 TiB (PR18)** | Dual-Region (`europe-west3` & `west4`) w/ `180d` $\rightarrow$ Coldline & `365d` $\rightarrow$ Delete lifecycle + Class A/B ops | `$5,420 / mo` | `$3,520 / mo` (`-35%`) |

## **Per-Shelf AI Inference Receipt & 12-Hour PT GSU Sizing (500,000 Shelves/Day)**

### **1. Per-Shelf AI Inference Receipt**

Grouping identical adjacent packs on a shelf row (`35` facings compressed into `~11` unique crop groups), matching `68%` of groups in memory with `gemini-embedding-2` (`-20%` Gen AI), and batching the remaining `32%` of lookalike ties (`~3.5` ties/shelf) into a single `gemini-3.5-flash-lite` contact sheet (`88.4%` context cache hit rate, `-20%` Gen AI v2) produces the following per-shelf and monthly AI inference cost at **500,000 shelves/day (15.0M shelves/month)**:

| **AI Inference Stage & Component** | **Volume per 35-Pack Shelf Photo** | **Standard PayGo (1.0x) / Shelf** | **EA Priority / 12h PT / Shelf** | **Monthly @ 500,000 Shelves/Day (15.0M/Mo)** | **Share of AI Cost** |
| --- | --- | :---: | :---: | :---: | :---: |
| **1. SKU Detection (PR06–PR08 Cloud Run CPU)** | `80 ms` 960px YOLO11n ONNX + adjacent pack grouping (`35` $\rightarrow$ `11`) | `$0.00003` (`₹0.003`) | `$0.00003` (`₹0.003`) | `$450 / mo` (`₹0.42L`) | `1.9%` |
| **2. Stage 2A gemini-embedding-2 (`-20%` Gen AI)** | `~11` unique crop embeddings (`68%` resolved immediately in RAM) | `$0.00050` (`₹0.046`) | `$0.00050` (`₹0.046`) | `$7,500 / mo` (`₹6.94L`) | `31.3%` |
| **3. Stage 2B gemini-3.5-flash-lite (`-20%` Gen AI v2, 32% ties)** | `~3.5` lookalike ties batched in 1 sheet (`88.4%` cached; `0` tok on `68%`) | `$0.00077` (`₹0.071`) | `$0.00105` (`₹0.097`) | `$15,750 / mo` (Std: `$11,550`) | `65.6%` |
| **4. Request Ingestion & Pub/Sub (`PR19`, `-25%` Pub/Sub)** | 1 JPEG write/read + Pub/Sub message + SQL result row | `$0.00002` (`₹0.002`) | `$0.00002` (`₹0.002`) | `$300 / mo` (`₹0.28L`) | `1.2%` |
| **Pure AI Inference Total (4 Workloads)** | **35-pack shelf $\rightarrow$ 5-Level SKU + Category SoS + P2P W1..W6 + Merch/Talker** | **`$0.00132 / shelf` (`₹0.12`)** | **`$0.00160 / shelf` (`₹0.15`)** | **`$24,000 / mo` (Std: `$19,800 / mo`)** | **`100.0%`** |

### **2. Daytime Store-Window Provisioned Throughput (PT) & Container Sizing (`07:30–18:00 IST`)**

Per Google Cloud's `gemini-3.5-flash-lite` Provisioned Throughput (PT) calculator, **1 GSU provides `3,550 tokens/sec` of throughput** (input + `4×` output weight).

At **500,000 shelves/day (15.0M/month)**, Stage 2B is called on `32%` of crop groups (`1` batched contact sheet per shelf on average, at `~1,450` cached/uncached input tokens + `45` JSON output tokens $\approx$ `1,630` weighted tokens per shelf):

- **Daytime store operating window (`07:30–18:00 IST` Mon–Fri; `07:30–19:00 IST` Sat–Sun, 10.5 active hours):** Processing `500,000 shelves/day` across the 10.5-hour store operating window averages **`13.2 shelves/sec`** (`~21,500 weighted tokens/sec`). Reserving **`6 to 8 GSUs` (`21,300 – 28,400 tok/s`)** covers steady daytime load, while midday peaks (`~33 shelves/sec`) spill over seamlessly onto PayGo without throttling.
- **Cloud Run CPU worker sizing (`10–12` active containers daytime; `25–40` midday peak):** At `2.69s` P50 shelf latency and `concurrency = 8` requests per `4 vCPU / 8 GiB` worker (`PR06–PR08`), steady daytime load (`13.2 shelves/sec`) requires **`10 to 12` active containers** (`25 to 40` containers at a `2.5×` midday burst, well inside `max 80`).
- **50% Half-Day Cost Attribution (`07:30–18:00 IST` Active / `18:00–07:30 IST` Shared Overnight):** Because store uploads run during the 10.5-hour daytime store operating window (`07:30–18:00 IST`), the AI Foundations team can route overnight batch jobs from other Unilever use cases onto those same `6–8` GSUs from `18:00` to `07:30 IST`. Any late rep uploads arriving after `18:00 IST` are served automatically via PayGo + scale-to-zero Cloud Run CPU workers without keeping a dedicated GPU cluster running overnight.

## **Shared Platform Overhead (`PR01–PR18` at 500,000 Shelves/Day)**

In `GCP_MarketShare_Configuration.xlsx` (`Prod` and `QA` sheets) and LLD001 Section 4.1.2, components `PR01` through `PR18` are shared across the retail platform.

The table below prices every `PR01–PR18` component at **500,000 shelves/day (15.0M/month)** under the contracted EA discount schedule (`20%` Cloud SQL Enterprise/Plus Compute & Storage, `25%` BigQuery/Composer/PubSub, `20%` Gen AI / v2), includes the **short-lived Cloud SQL for PostgreSQL vector catalog DB**, and retires `PR13` Document AI (`218,300` pages/mo in `GCP_MarketShare_Configuration.xlsx` and `2,183,900` pages/mo = `85k` pages/day in Azure PROD `6712-Marketshare-Form-Recognizer` for Toker OCR, now handled natively by Stage 2B):

| **Component ID & GCP Service (GCP_MarketShare_Configuration.xlsx)** | **Production + QA Sizing & Applied EA Discount (europe-west3 & west4)** | **Monthly @ 500,000 Shelves/Day (15.0M/Mo)** | **Unit Cost / Shelf ($ / ₹)** | **Share of All-In EA Spend** |
| --- | --- | :---: | :---: | :---: |
| **1. Stage 2B gemini-3.5-flash-lite (`PR14`, `-20%` Gen AI v2)** | `32%` lookalike ties + Promo/Talker OCR (12h shared PT + PayGo spillover; replaces `PR13` Document AI) | **`$15,750 / mo`** (Std: `$11,550`) | `$0.00105` (`₹0.097`) | `43.3%` |
| **2. Stage 2A gemini-embedding-2 (`PR14`, `-20%` Gen AI)** | `~11.1` unique crop groups/shelf across 15.0M photos/mo (`68%` fast exit in RAM) | **`$7,500 / mo`** | `$0.00050` (`₹0.046`) | `20.6%` |
| **3. Cloud SQL Enterprise / Plus (`PR15–PR17`, `-20%` Cloud SQL)** | `PR15` Processing (`16 vCPU, 64 GB, 1 TB`), `PR16` MT & `PR17` GT (`16 vCPU, 128 GB, 1 TB` each) + QA (`3× 2 vCPU`) + Short-Lived Cloud SQL PostgreSQL Vector DB (`$8,120` list $\times 0.80$) | **`$6,500 / mo`** (fixed, `-20%`) | `$0.00043` (`₹0.040`) | `17.9%` |
| **4. Cloud Storage 300 TiB (`PR18` Prod + 10 TiB QA)** | Dual-region (`europe-west3/4`) w/ Azure-parity lifecycle (`180d` $\rightarrow$ Coldline, `365d` $\rightarrow$ Delete) + Class A/B ops | **`$3,520 / mo`** (fixed) | `$0.00023` (`₹0.021`) | `9.7%` |
| **5. Cloud Run SKU Detection (`PR06–08`) + Pub/Sub (`PR19`, `-25%`) + App (`PR03–05`)** | `PR06–08` Inference workers (`4 vCPU / 8 GiB, min 4 / max 80`) + `PR19` Pub/Sub (`-25%`) + `PR03–05` Frontend/Backend Annotation & App pods + `0.2%` Judge | **`$1,850 / mo`** (`$750` AI + `$1,100` App) | `$0.00005` AI + `$0.00007` App | `5.1%` |
| **6. Shared Load Balancers (`PR01–02`), Dataflow (`PR12`), Composer (`PR11`, `-25%`) & BigQuery (`-25%`)** | External + Internal Regional ALB (`PR01–02`), Secret Mgr (`PR09`), Artifact Reg (`PR10`), Cloud Composer (`1` scheduler, `PR11` at **`-25%`**), BigQuery (`-25%`), Dataflow (`240h × 3` nodes, `PR12`) | **`$1,280 / mo`** (fixed) | `$0.00009` (`₹0.008`) | `3.5%` |
| **7. Document AI (`PR13`: 218.3K GCP / 2.18M Azure Toker pages/mo)** | **Eliminated (`₹0`):** Stage 2B `gemini-3.5-flash-lite` + `talker_price_tag_detect.py` replace Azure Form Recognizer (`85k` pages/day) | **`$0.00`** (saves `$2,600–$3,275/mo`) | `₹0.000` | `0.0%` |
| **Subtotal A: Pure AI Inference Pipeline Only (4 Workloads)** | **Covers `PR06–PR08` SKU Detection + `PR14` Vertex AI Embeddings & Flash-Lite (`-20%` Gen AI/v2) + `PR19` Pub/Sub (`-25%`) + `0.2%` Judge** | **`$24,000 / mo` (Std: `$19,800 / mo`)** | **`$0.00160 / shelf` (`₹0.15`)** | **`65.9%`** |
| **Subtotal B: Shared Platform & DB Overhead (`PR01–05`, `PR09–18`)** | **Covers Cloud SQL (`PR15–17` at `-20%`), 300 TiB GCS (`PR18`), App/Annot (`PR03–05`), LB/Dataflow & Composer/BQ (`PR01–02, 11–12` at `-25%`)** | **`$12,400 / mo` (fixed `PR01–PR18`)** | **`$0.00083 / shelf` (`₹0.08`)** | **`34.1%`** |
| **All-In Portfolio Total @ 500,000 Shelves/Day (15.0M/Month)** | **Complete AI Pipeline + All `PR01–PR18` Shared Infrastructure (Prod + QA)** | **`$36,400 / mo` (Std: `$32,200 / mo`)** | **`$0.00243 / shelf` (`₹0.22`)** | **`100.0%`** |

## **Comparing Architecture Options at 500,000 Shelves/Day (15.0M/Month)**

Note: Flex Batch (`0.5x`) is excluded from production serving because asynchronous batch queueing does not meet interactive store callback SLAs (`PT180S`). All options share the identical `$12,400 / mo` (`PR01–PR18`) platform & database overhead under the contracted `20%` Cloud SQL and `25%` Composer/BigQuery/PubSub discounts:

| **Architecture Option** | **Serving Tier & Escalation (`t.ire4ghmqzbz8` Validated)** | **5-Attr SKU Acc (5K Holdout)** | **Pure AI / Shelf ($ / ₹)** | **Pure AI Monthly @ 500K/Day (15M/Mo)** | **All-In Monthly w/ $12.4K PR01–18 @ 500K/Day** |
| --- | --- | :---: | :---: | :---: | :---: |
| **Option 2 + Local CPU ONNX Embedder** | Distilled CPU ONNX Embedder (`$0` API) + `gemini-3.5-flash-lite` (`32%` ties, `-20%` Gen AI v2) | `82.1% – 85.0%` | `$0.00058` (`₹0.06`) | **`$8,700 / mo`** | **`$21,100 / mo`** |
| **Option 1: Embedding-Only (`clm_standalone`)** | `gemini-embedding-2` (`0` VLM calls; `-20%` Gen AI; `1013-013247` / `1016-092457` budget mode) | `78.9% – 81.2%` | `$0.00105` (`₹0.10`) | **`$15,750 / mo`** | **`$28,150 / mo`** |
| **Option 2: Hybrid (Standard 1.0x + 12h PT)** | **`gemini-embedding-2` + `gemini-3.5-flash-lite` Standard / 12h PT (`32%` ties, `-20%` Gen AI/v2, `1016-092456`)** | **`82.3% – 84.0%`** | **`$0.00132` (`₹0.12`)** | **`$19,800 / mo`** | **`$32,200 / mo`** |
| **Option 2: Hybrid (EA Priority / PT Ceiling)** | **`gemini-embedding-2` + `gemini-3.5-flash-lite` Priority (`32%` ties, `-20%` Gen AI/v2, `1021-154455`)** | **`82.3% – 84.0%`** | **`$0.00160` (`₹0.15`)** | **`$24,000 / mo`** | **`$36,400 / mo`** |
| **Option 2 Stress Cap (50% Ties)** | Stress test: `50%` VLM tie-break escalation (`-20%` Gen AI/v2) | `82.3%+` | `$0.00219` (`₹0.20`) | `$32,850 / mo` | `$45,250 / mo` |
| **Existing 14-Model GPU Stack (`cv_models.py`)** | `14× g2-standard-8` L4 GPUs (`-23%` Standard AI Platform: `$28.8K` Prod / `$38.6K` w/ QA) + `PR13` + DBs | `42.32%` (`1016-093317`) | `$0.00192 – $0.00257` | `$28,800 – $38,600 / mo` | `$41,200 – $51,000 / mo` |
| **Self-Hosted Open-Weight GPUs (vLLM)** | `24× NVIDIA L4` GPUs (`-23%` Standard AI Platform), idle overnight outside `07:30–18:00 IST` | `~81.0% – 83.0%` | `$0.00327` (`₹0.30`) | `$49,100 / mo` (`24× L4` GPUs) | `$61,500 / mo` |
