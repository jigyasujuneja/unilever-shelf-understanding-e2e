# Unilever Shelf Understanding: Datasets Guide & Quick-Start README

This one-pager explains the datasets in [`data/`](../data/) (and `gs://marketshare-datasets/derived_manifests/`), the design thinking behind them in plain English, which dataset to pick for each engineering task, and how to combine `gallery + train` when fine-tuning a supervised classifier.


## 1. The Thinking Behind Our Data Design (In Plain English)

### Why We Did Not Train or Evaluate Directly on Raw GCS Folders
The raw client buckets (`gs://marketshare-datasets`, `gs://gtmerch-marketshare`, and `gs://ms-json-dataset`) contain over `1,002,209` cropped images and `53,019` shelf JSONs across separate folder trees (`brand_data`, `variant_*`, `pack_data`). Auditing the raw files exposed four issues that distort both training and evaluation if used as-is:
1. **Cross-Silo Duplicates (`399,769` duplicate files)**: The exact same cropped image bytes appear across `brand_data`, `variant_*`, and `pack_data`. A random split puts identical store crops in both `train` and `test`.
2. **In-Place Synthetic Copies (`8,398+` files)**: Rare folders were padded with `copy*.jpg`, `(copy)`, and `DA_TBC_*` jittered crops (`78%` of `Oral_Care__Can` and `66%` of `Beverages__Tin` are copies).
3. **Multi-Folder Label Conflicts (`554` MD5 hashes)**: Byte-identical crops sit in two competing folders simultaneously (such as `Bottle` vs. `Jar`, or base pack vs. `Rs 10` promo pack).
4. **Burned-In Debug Overlays (`100%` of `mt/` and `33%` of `gt/` in `ms-json-dataset`)**: Red OpenCV bounding boxes and green text labels were drawn directly into the raw image pixels before saving.

### Why Every Dataset Uses the `classes` + `gallery` + `splits` Structure (`rpc.json`)
Instead of baking 2,659 SKU classes into static ResNet weights that require retraining whenever packaging changes, our production architecture treats SKU recognition as **visual search + VLM verification** (matching shelf crops against a reference catalog in AlloyDB / Vector Search).

To support both **visual retrieval pipelines** and **supervised fine-tuning (SFT)** from a single source of truth, every `rpc.json` manifest contains three sections:
1. **`classes` (Master Product Dictionary)**:
   Maps each integer `sku_id` (`"1"` to `"2659"`) to its 5-level canonical hierarchy:
   * `is_hul` (`true` for Unilever brands including Horlicks, Boost, V-Wash, Indulekha, Glow & Lovely; `false` for competitors)
   * `category` (`10` store categories: `Skin`, `Hair_Care`, `PW`, `Oral_Care`, `DMT`, `Laundry`, `HHC`, `HFD`, `Beverages`, `Foods`)
   * `brand` (`934` canonical brands)
   * `base_variant` (`Brand__Variant`, separating visual identity from promo callouts)
   * `packaging_type` (`16` physical forms: `Bottle`, `Pouch`, `Tube`, `Jar`, `Bar`, `Carton`, `Sachet`, `Wrapper`, `Tub`, `Can`, `Tetrapack`, `Pump`, `Roll_On`, `Tin`, `Box`, `Tray`, with `0` `Unknown` labels)
2. **`gallery` (Reference Catalog Index - `100%` Class Coverage)**:
   Holds `1` to `4` clean reference crop URIs for **every class in `classes`** (`10,082` images across `2,659` classes in `hul_golden`; `7,442` images across `1,922` classes in `hul_variants`), with **zero overlap with `val` or `test`**.
   * *How retrieval/hybrid pipelines use `gallery`*: During `setup()`, the pipeline embeds `gallery` images into the vector index. At test time, System 1 matches each query crop against `gallery`; if top-4 candidates are close, System 2 builds a `2x2` visual contact sheet from the `gallery` JPEGs for Gemini verification.
3. **`splits` (`train`, `val`, `test` - Disjoint by Store Visit `session_id`)**:
   * **`val` and `test`**: Held-out query images from separate physical store visits (`1,253` store visits in `val`, `1,226` in `test`, with `0` shared store visits). Stratified `~50%` HUL / `~50%` Non-HUL across all 10 categories and packaging forms.
   * **`train` (`train.jsonl`)**: Additional store crops left over after reserving `gallery`, `val`, and `test`.


## 2. Quick Reference: All Datasets & Which One to Use

| Dataset Path | Catalog Size | Gallery Images | Train / Val / Test Counts | Primary Use Case & When to Use It |
| :--- | :--- | :--- | :--- | :--- |
| **[`data/hul_golden/`](../data/hul_golden/rpc.json)**<br>`gs://.../derived_manifests/hul_golden/` | **`2,659` classes**<br>(`1,049` HUL, `1,610` Comp) | `10,082`<br>(`100%` classes) | `15,989` / `1,900` / `1,905` crops | **Primary Stage 2 Crop Benchmark & Multi-Task SFT**. Covers all `1,922` specific `Brand__Variant` SKUs plus `737` brand-level competitor/HUL classes. Use this when your model must output all 5 hierarchy levels (`is_hul`, `category`, `brand`, `base_variant`, `packaging_type`) across the full store catalog. |
| **[`data/hul_variants/`](../data/hul_variants/rpc.json)**<br>`gs://.../derived_manifests/hul_variants/` | **`1,922` classes**<br>(`1,048` HUL, `874` Comp) | `7,442`<br>(`100%` classes) | `15,661` / `1,900` / `1,905` crops | **Pure `Brand__Variant` Classifier & Retrieval Benchmark**. Removes the `737` brand-only classes and `<Brand>__Others` catch-all folders so `100%` of classes are explicit `Brand__Variant` SKUs (`1,048` HUL from `variant_*` + `874` Non-HUL from `variant_NH_*`). Use this for fine-grained variant classification. |
| **[`data/hul_brands/`](../data/hul_brands/rpc.json)**<br>`gs://.../derived_manifests/hul_brands/` | **`2,659` classes**<br>(`934` brands) | `10,082` | `8,000` / `1,000` / `1,000` crops | **Brand & M&A Ownership Attribution**. 50/50 HUL vs. Non-HUL stratified subset of `hul_golden`. Use this to benchmark brand accuracy and HUL vs. competitor ownership (`is_hul`). |
| **[`data/hul_packs/`](../data/hul_packs/rpc.json)**<br>`gs://.../derived_manifests/hul_packs/` | **`2,659` classes**<br>(`16` pack types) | `10,082` | `6,000` / `1,000` / `1,000` crops | **Physical Container Classification**. Stratified across all 16 packaging types with `0` `Unknown` labels and multi-modal `allowed_packaging_types` support. |
| **[`data/hul_promos/`](../data/hul_promos/rpc.json)**<br>`gs://.../derived_manifests/hul_promos/` | **`2,659` classes**<br>(`179` promo SKUs) | `10,082` | `4,000` / `1,000` / `1,000` crops | **Promotional & Price-Point Callout Detection**. Evaluates `promo_type` (`B4G1`, `RS10`, `RS5`, `PACK_OF_2`, `FREE_COFFEE_MUG`) on promotional packs. |
| **[`data/ms_json_golden/`](../data/ms_json_golden/rpc.json)**<br>`gs://.../derived_manifests/ms_json_golden/` | **`2,659` classes** | `10,082` | `140` / `60` / `100` shelves<br>(`4,859` / `2,017` / `3,709` boxes) | **Primary Full-Shelf End-to-End (E2E) Benchmark**. Curated from `gs://ms-json-dataset/gt/` after rejecting all images with burned-in OpenCV boxes. Use this to evaluate Stage 1 box detection + Stage 2 multi-SKU identification + Share-of-Shelf (SOS) on real uncropped store photos. |
| **[`data/gtmerch_golden/`](../data/gtmerch_golden/rpc.json)**<br>`gs://.../derived_manifests/gtmerch_golden/` | **`2,659` classes** | `10,082` | `400` / `100` / `200` shelves<br>(`14,692` / `3,824` / `8,070` boxes) | **Stage 1 Dual-Class Display Bay (`Class 0`) vs. Product (`Class 1`) Detection & Bay Isolation**. Built from `gs://gtmerch-marketshare`. Use this to evaluate separating branded promotional headers/coolers (`asset_boxes`) from product facings (`boxes`) and verifying in-bay brand purity. |
| **[`data/hul_sessions/`](../data/hul_sessions/rpc.json)**<br>*(Alias: `data/gt_hul_sessions/`)* | **`2,659` classes** | `10,082` | `150` / `50` / `100` strips<br>(`1,200` / `400` / `800` boxes) | **Multi-Facing Store Session Retrieval & Row Smoothing**. Groups `8` verified crops from the same store visit (`session_id`) into a `1760x260` strip. Use this to test Stage 2 medoid deduplication, 1D row smoothing (`[P, ?, P] -> P`), and session SOS (do not use for Stage 1 box detection). |


## 3. How to Train a Supervised Classifier (SFT) on `hul_variants` or `hul_golden`

### Why You Should Combine `rpc.json["gallery"]` + `splits["train"]` for Closed-Set Training
When training a closed-set classifier from scratch (rather than running retrieval against `gallery`), **do not train on `splits["train"]` (`train.jsonl`) alone**:
* In `hul_variants`, `356` rare long-tail classes have between `1` and `4` clean images total in the bucket. The manifest builder placed those `1` to `4` images into `rpc.json["gallery"]` so every class has a reference anchor, leaving `1,502` classes in `train.jsonl`.
* Because **neither `rpc.json["gallery"]` nor `splits["train"]` overlaps with `val` or `test` (`0` shared URIs or MD5s)**, you should take the **union of `rpc.json["gallery"]` and `splits["train"]`** as your supervised training set:

| Dataset | `splits["train"]` Alone | Union of `rpc.json["gallery"] + splits["train"]` (Deduplicated) | `val` Split | `test` Split |
| :--- | :--- | :--- | :--- | :--- |
| **`hul_variants`** (`1,922` classes) | `15,661` images (`1,502 / 1,922` classes) | **`22,104` unique images (`1,922 / 1,922` = `100%` of classes)** | `1,900` images (`0` overlap) | `1,905` images (`0` overlap) |
| **`hul_golden`** (`2,659` classes) | `15,989` images (`1,516 / 2,659` classes) | **`25,071` unique images (`2,659 / 2,659` = `100%` of classes)** | `1,900` images (`0` overlap) | `1,905` images (`0` overlap) |

### 10-Line Python Snippet to Build the Complete `1,922`-Class Training Split (`22,104` Images)
```python
import json

with open("data/hul_variants/rpc.json") as f:
    mf = json.load(f)

classes = mf["classes"]  # 1,922 Brand__Variant classes (or use data/hul_golden/rpc.json for 2,659 classes)
train_by_uri = {}

# 1. Add all gallery reference crops (covers 100% of classes: 1,922 / 1,922)
for cid, uris in mf["gallery"].items():
    for u in uris:
        train_by_uri[u] = {"uri": u, "sku_id": int(cid), "label": classes[cid]}

# 2. Merge splits["train"] (deduplicating the 999 shared URIs; 0 overlap with val or test)
for row in mf["splits"]["train"]:
    cid = str(row["products"][0])
    train_by_uri[row["uri"]] = {"uri": row["uri"], "sku_id": int(cid), "label": classes[cid]}

full_train_records = list(train_by_uri.values())
print(f"Total unique train images: {len(full_train_records)} across {len({r['sku_id'] for r in full_train_records})} classes")
# Output: Total unique train images: 22104 across 1922 classes
```

### Pre-Built SFT JSONL Files Available in GCS
If you are running Vertex AI / Gemini Supervised Fine-Tuning or need the uncapped raw HUL corpus (`381,619` crops across `1,104` HUL folders), use these ready-made JSONLs in `gs://marketshare-datasets`:
1. **Multi-Task 5-Attribute Hierarchy SFT (`hul_golden`)**:
   * `gs://marketshare-datasets/derived_manifests/hul_golden/sft/sft_multitask_train.jsonl`
   * `gs://marketshare-datasets/derived_manifests/hul_golden/sft/sft_multitask_val.jsonl`
2. **Per-Folder Stratified HUL Variant SFT (`<= 50` train / `<= 10` val per folder, built by [`src/utils/tuning.py`](../src/utils/tuning.py))**:
   * `gs://marketshare-datasets/input_data/jsonl/train_variant_stratified.jsonl`
   * `gs://marketshare-datasets/input_data/jsonl/val_variant_stratified.jsonl`
3. **Uncapped Raw HUL Variant Corpus (`381,619` HUL crops across `1,104` folders)**:
   * `gs://marketshare-datasets/input_data/jsonl/train.jsonl`


## 4. Important Rules & Nuances to Know Before Running Experiments

1. **Why `1,104` vs. `1,922` vs. `2,659` Classes Exist**:
   * **`1,104`** is the raw folder count inside the 5 HUL-only variant subdirectories (`variant_skin`, `variant_wash`, `variant_p5`, `variant_gen`, `variant_oral`), which includes `56` `<Brand>__Others` catch-all folders and `0` competitors.
   * **`1,922` (`hul_variants`)** is the de-collided `Brand__Variant` catalog combining **`1,048` HUL variants** (after removing `<Brand>__Others`) + **`874` Non-HUL competitor variants** from `variant_NH_gen` and `variant_NH_P6`.
   * **`2,659` (`hul_golden`)** adds the **`737` brand-level competitor and HUL classes** from `brand_data` + `pack_data` on top of the `1,922` `Brand__Variant` classes so open-world shelf crops of any brand can be classified.
2. **Understanding `is_others_catchall` on Full Shelves (`ms_json_golden` & `gtmerch_golden`)**:
   * All crop datasets (`hul_golden`, `hul_variants`, `hul_brands`, `hul_packs`, `hul_promos`, `hul_sessions`) have **`0` `is_others_catchall` boxes**. Every crop has a verified catalog label.
   * On full uncropped shelves in `ms_json_golden`, human annotators labeled `69.3%` of boxes with an exact `Brand__Variant` and `26.7%` of boxes (unknown local/regional rival products) as `Others____Others____NA` (`is_others_catchall = True`). We keep those boxes so Stage 1 detectors are not penalized for detecting real products on the shelf and so the Share-of-Shelf denominator (`HUL facings / Total shelf facings`) is exact. For `is_others_catchall` boxes, only `is_hul == False` is known.
   * On `gtmerch_golden`, raw annotations only had `Class 0` (promo bay) and `Class 1` (product facing) bounding boxes (`0` human per-box SKU labels). Use `gtmerch_golden` for **Stage 1 Bay vs. Product Detection** and **In-Bay Brand Compliance**, and use `hul_golden` / `hul_variants` / `ms_json_golden` for fine-grained SKU accuracy.
3. **`hul_sessions` and `gt_hul_sessions` Are the Same Dataset**:
   * Both point to the same `150` train, `50` val, and `100` test 8-crop horizontal session strips (`1760x260` px with 16px gray gutters). Use `hul_sessions` to evaluate **Stage 2 multi-facing retrieval, medoid deduplication, 1D shelf-row smoothing, and session SOS**. Do not use `hul_sessions` as a Stage 1 box detection benchmark (use `sku110k`, `ms_json_golden`, and `gtmerch_golden` for Stage 1 detection).
4. **Deterministic Subsampling via [`sample_images()`](../src/utils/dataset.py#L222-L260)**:
   * Passing `--limit 50 --seed 0` (for fast screening) or `--limit 1000 --seed 0` / `--limit 0` (for full evaluation with `±1.3%` 95% confidence intervals) selects the exact same interleaved `50% HUL / 50% Non-HUL` subset on every machine, making all runs directly comparable.


## 5. Client Discovery & Sign-Off Questions: GT Data, `Others` Catch-Alls, Unmapped Classes & Production Usage

Use these targeted, evidence-backed questions when reviewing the data and legacy `.h5` pipeline with the Client AI Lead and Unilever Data Engineering team:

### A. General Trade (GT) Merchandising & Session Data (`gs://gtmerch-marketshare` & `gs://ms-json-dataset/gt/`)
1. **Why do raw `gtmerch-marketshare` annotations only contain `Class 0` (Display Bay) and `Class 1` (Product Facing) boxes with `0` per-box SKU labels?**
   * *Context*: In `gs://gtmerch-marketshare`, `92.1%` of `Class 1` product boxes sit *outside* the branded `Class 0` HUL display bay on the surrounding Kirana counter/wall, and only the image filename carries a campaign brand code (such as `495-PPVWASHHANUCMOC4AD.jpg`).
   * *Question*: **In your live GT Merchandising system, do you only score `Class 1` boxes inside the `Class 0` HUL display bay (to measure bay purity and flag foreign competitor intruder packs), or do you also compute total store Share-of-Shelf across the `92.1%` of unannotated counter products outside the bay?**
2. **Do you have the Master Campaign-Code-to-Planogram lookup table for `gtmerch-marketshare`?**
   * *Question*: **Can you share the mapping table that links filename campaign codes (`PPVWASH...`, `PPLUX...`, `PPDOVE...`) to the exact target `Brand__Variant` SKUs required in each GT display bay?**
3. **Uncropped Full-Shelf Parent Images for `gt_hul_sessions` / `variant_*` Crops**:
   * *Context*: The `381,619` `variant_*` crops and `98,488` `pack_data` crops are isolated cutouts without full-shelf parent photos, and `100%` of `ms-json-dataset/mt/` (`+ 33%` of `gt/`) have red OpenCV boxes and green text permanently burned into the JPEG pixels.
   * *Question*: **Can your data engineering team export the un-annotated raw parent shelf photos (before `cv2.rectangle` overlays were drawn) and the bounding-box coordinate CSVs for the `340xxxx` and `HUL-*` store sessions?**

### B. How the Client Uses `"Others"` & `"Not Required"` Across Training and Production
1. **Full-Shelf `Others____Others____NA` (`26.7%` of all boxes in `gs://ms-json-dataset/gt/`)**:
   * *Context*: Human annotators labeled `26.7%` of full-shelf boxes (`991 / 3,709` in `test`) as `Others____Others____NA` and `3.9%` as `<Brand>____Others____<Pack>`.
   * *Question*: **Does `Others____Others____NA` strictly mean "Verified Non-HUL Regional Competitor", or did annotators also use it for blurry/unreadable HUL packs?** *(If it is strictly Non-HUL, those boxes belong in the Share-of-Shelf denominator `HUL / Total Facings`; if it includes unreadable HUL packs, including them artificially depresses HUL Share-of-Shelf.)*
2. **Variant-Tree `<Brand>__Others` (`61` folders) and `<Brand>__<Brand>` (`59` folders)**:
   * *Context*: Raw `variant_*` contains `34` HUL `<Brand>__Others` folders, and `variant_NH_*` contains `27` `<Brand>__Others` and `59` `<Brand>__<Brand>` folders (`Dermaco__Dermaco`, `Lizol__Lizol`). Similarly, the 7 legacy GT `.h5` variant models include `"Others"` and `"Not Required"` as explicit softmax output classes.
   * *Question*: **When a model predicts `<Brand>__Others` or `"Not Required"` in production, how does business reporting treat that facing?** Is it credited toward **Brand Share-of-Shelf (`L2 Brand`)** while excluded from Variant compliance, or is it dropped?
   * *Question*: **Do you agree with removing `<Brand>__Others` as a trainable softmax class?** Training a classifier on a visual catch-all folder teaches it to route slightly blurry crops of *known* variants into `"Others"`. Instead, our pipeline keeps the catalog 100% clean (`1,922` explicit `Brand__Variant` classes) and uses a calibrated **cosine/margin threshold gate** to back off to Brand-level (`L2`) only when a crop is genuinely unreadable.

### C. Unmapped & Orphan Classes Across Silos and Legacy `.h5` Models
1. **Why do the 7 legacy GT `.h5` variant models only cover 7 brand families (`79` variant classes) out of `934` brands and `1,922` variants?**
   * *Context*: In `cv_classify`, `brand.h5` predicts `59` brands, and only `7` GT brand families route to a downstream variant `.h5` model (`79` total variant classes). Even with **100% Oracle Ground-Truth Brand routing** (`1015-115810`), the `.h5` pipeline caps at `52.0%` 5-Attr accuracy (`44.5%` end-to-end).
   * *Question*: **How does your current production pipeline classify variants for the other `52` HUL brands in `brand.h5` (and the `~1,025` HUL variant folders not covered by the 7 GT `.h5` models)?** Do you run separate category-level `.h5` models for MT, or does production stop at Brand + Pack Type for those brands?
2. **How does `brand.h5` reject Non-HUL competitor products in the wild?**
   * *Context*: All `59` output classes of `brand.h5` map to `is_hul = True` (`50.0%` L1 HUL accuracy on balanced HUL/competitor splits), whereas `brand_data` on GCS contains `874` Non-HUL competitor brand folders (`AColgate`, `ADabur`, `AGodrej`, `Amul`, etc.) that have zero folders in `variant_*`.
   * *Question*: **In production, how do you prevent `brand.h5` from hallucinating HUL brands on the `50%` to `70%` of shelf facings that belong to competitors?** Do you have a separate binary HUL-vs-Competitor gate model upstream of `brand.h5`, or a softmax confidence threshold?
3. **Meaning of `A`-Prefix Folders (`AColgate`, `ADabur`, `ABrook_Bond`, plus `7` Twin Pairs like `ALux` / `Lux`) and 1-2 Letter Folders (`W2`, `Z`, `An`, `Fr`, `He`) in `brand_data`**:
   * *Question*: **Was the `A` prefix added by annotators to sort priority brands to the top of the folder list?** Can we permanently merge the `7` `A`-prefix twin pairs (`ADove`/`Dove`, `ALux`/`Lux`, etc.), and should the `3` HUL brand-only folders (`Axe`, `Brut`, `ABrook_Bond`, which have `0` folders in `variant_*`) be scored at the Brand + Pack level only?

### D. Cross-Silo Linkage, Label Conflicts & New SKU Onboarding
1. **Cross-Silo Filename Collisions (`20,421` shared `<session>_<crop>.jpg` names, `21.4%` different images)**:
   * *Context*: Because the crop counter (`_0.jpg, _1.jpg`) reset across category runs on the same store photo, `1098_53.jpg` is `AAyush` in `brand_data` and `Rexona__Sport_Defence_Roll_On` in `variant_gen`. Meanwhile, `399,769` crops are byte-identical MD5 duplicates across `brand_data`, `variant_*`, and `pack_data`.
   * *Question*: **When your team links `brand_data`, `variant_*`, and `pack_data`, are you joining by exact `md5_hash` rather than filename?** *(Joining by filename injects a `21.4%` cross-silo label corruption rate.)*
2. **Resolving the `554` Multi-Folder MD5 Conflicts (`497` in `variant_*`, `57` in `pack_data`)**:
   * *Question*: **What is Unilever's official catalog rule separating `Bottle` vs. `Jar` in Health Food Drinks (`Horlicks`), `Pouch` vs. `Sachet` on `Rs 5` / `Rs 10` packs (`Boost`), and should promotional callout folders (`_B3G1`, `_PO2`, `_Rs_10`) roll up into their parent `base_variant` for Market Share while being scored separately on `promo_type` for Merchandising?**
3. **Day-1 Cold-Start Onboarding for New Launches & Rare Tail SKUs (`356` classes with `1-4` images)**:
   * *Question*: **When Unilever launches a new variant or promo pack with only `1` to `4` studio renders (`HUL-*`), how do you onboard it without waiting weeks to collect `100+` store crops for softmax retraining?** *(In our 2-Stage Hybrid pipeline, dropping `1-4` studio renders into `rpc.json["gallery"]` makes a new SKU live in `< 60 seconds` with zero retraining; for supervised training, merging `rpc.json["gallery"] + splits["train"]` immediately provides `22,104` leak-free images across `100%` of the `1,922` `hul_variants` classes.)*


## 6. Operational Architecture Comparison & Top 3 MVP Shortlist for A/B Testing

When selecting which architectures to transfer to the MVP stage and field A/B testing, aggregate test accuracy on a static split is only half of the decision criteria. **Day-2 operational viability** (new SKU onboarding speed, rare regional SKU performance, open-set competitor rejection on wild shelves, and MLOps retraining burden) determines whether a pipeline holds up in live retail operations.

### 6A. Forensic Stratification by Training Sample Count (`hul_golden` `test`, $N=1,000$)

Stratifying the `1,000` held-out test crops of `hul_golden` by how many labeled shelf crops each SKU had in `splits["train"]` (`15,989` crops across `1,538` classes out of `2,659` total catalog classes) exposes why closed-set supervised classifiers (`cv_classify` and `dinov3_classify`) fail on the long tail and on newly launched SKUs:

| SKU Frequency in `hul_golden/train` | Share of Test Split ($N=1,000$) | **Standalone `DINOv3/v2` SFT (`#113613`)** | **CLM `gemini-embedding-2` zero-generative-VLM (`#013247`)** | **2-Stage Hybrid `Gemini 3.5 Flash-Lite` (`#025000`)** | Operational Implication |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`16+` train crops** (Head SKUs) | `59.5%` ($595$ crops) | **`90.9%`** ($541/595$) | `75.2%` | `78.1%` | Closed-set SFT memorizes the ~150 highest-volume SKUs that have dozens of shelf crops in `train`. |
| **`6 to 15` train crops** (Mid-tier SKUs) | `22.9%` ($229$ crops) | `70.7%` ($162/229$) | `84.2%` | **`89.5%`** | Below 16 training crops, closed-set SFT drops **`18.8 pp` behind** the 2-Stage Hybrid. |
| **`3 to 5` train crops** (Low-tier SKUs) | `9.7%` ($97$ crops) | `56.7%` ($55/97$) | **`89.9%`** | **`88.9%`** | With 3 to 5 training crops, closed-set SFT drops **`32.2 pp` behind** the 2-Stage Hybrid. |
| **`1 to 2` train crops** (Rare / Regional SKUs) | `6.1%` ($61$ crops) | **`27.9%`** ($17/61$) | `74.6%` | **`79.1%`** | With 1 to 2 training crops, closed-set SFT collapses to **`27.9%`** (**`51.2 pp` behind** Hybrid). |
| **`0` train crops** (New Launches / Tail SKUs) | `1.8%` ($18$ crops) | **`0.0%`** ($0/18$) | `73.1%` | **`80.8%`** | **`1,121` of `2,659` catalog SKUs (`42.2%`) have `0` rows in `splits["train"]`**. Closed-set SFT scores **`0.0%`**, whereas the gallery-driven Hybrid scores **`80.8%`** zero-shot. |

### 6B. Six-Dimension Operational Decision Matrix

| Operational Requirement | Legacy 9-Model `.h5` Chain (`cv_classify`) | Standalone `DINOv3/v2` SFT (`dinov3_classify`) | **MVP Arm A: `CLM Standalone` (`gemini-embedding-2` zero-generative-VLM)** | **MVP Arm B (Primary Winner): `2-Stage Hybrid` (`CLM + Gemini 3.5 Flash-Lite`)** | **MVP Arm C (Max-Precision SFT): `Hybrid + LoRA SFT Gemini 3.5 Flash-Lite`** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. `hul_golden` / `hul_variants` Accuracy (3-Attr / 5-Attr)** | `51.0%` / `44.5%` *(Disqualified)* | `88.2%` / `77.5%` *(Head-skewed)* | **`89.5%` / `78.6%`** | **`91.6%` / `82.6%`** *(Zero-shot)* | **`96.7%` L4 / `93.5%` L8** *(Fine-tuned)* |
| **2. New SKU Launch / Promo Onboarding (`0` shelf training crops)** | **`0.0%`** *(Requires full `.h5` retraining)* | **`0.0%`** *(Fixed `1,538`-row weight matrix; `1,121` catalog SKUs missing)* | **`73.1%`** *(Add `1` reference image to `hul_gallery`, live in `<60s`)* | **`80.8%`** *(Add `1` reference image to `hul_gallery`, live in `<60s`)* | **`80.8%+`** *(Gallery handles new SKUs; LoRA SFT boosts established lines)* |
| **3. Rare & Regional Long-Tail SKUs (`1-2` shelf crops)** | `< 15.0%` | **`27.9%`** *(Overfits to `16+` crop head SKUs)* | **`74.6%`** | **`79.1%` (`+51.2 pp` over DINOv3)** | **`88.0%+`** |
| **4. Open-Set Competitor Rejection (`Other` on Wild Shelves)** | **Fails (`50.0%` L1 HUL)** *(Predicts HUL on 100% of crops)* | **Fails** *(No abstention threshold; forces `argmax` into 1,538 classes)* | **Passes (`94.9%` L1 HUL on `gtmerch_golden`)** | **Passes (`95.2%` L1 HUL, `1.88%` SOS error on `gtmerch_golden`)** | **Passes (`98.4%` L1 HUL)** |
| **5. Fine-Print Pack Size (`50g` vs `100g`) & Promo (`B4G1`, `Rs 10`) OCR** | **No** *(224px CNN)* | **No** *(224px patch average)* | **Partial** *(Visual embedding only)* | **Yes (`88.3%` on `hul_promos`, bottom-30% Zoom-Strip OCR)** | **Yes (`93.5%` L8 E2E)** |
| **6. Cost per Crop & Day-2 MLOps Maintenance** | `₹0.0035` compute + **High MLOps** (9 separate `.h5` models to retrain) | `₹0.0022` compute + **High MLOps** (Must collect & label `16+` shelf crops per new SKU) | **`₹0.0219`** (`0` VLM calls, **Zero Retraining**) | **`₹0.0538`** (`59.1%` zero-generative-VLM gate, **Zero Retraining**) | **`₹0.0365`** (`europe-west4` LoRA endpoint + quarterly SFT refresh) |

### 6C. Recommended Top 3 MVP Shortlist for Production Transfer & A/B Testing

1. **MVP Arm A (Ultra-Low-Cost / High-Throughput Baseline): `960px Soft-NMS ONNX + CLM Standalone (gemini-embedding-2, zero-generative-VLM)`**
   * **Role in MVP**: Zero generative VLM calls (`₹0.022/crop`, `0.21s` p50), delivering **`89.5%` 3-Attribute (`Brand + Category + Pack`) accuracy** (`98.6%` HUL vs Competitor) and **`80.7%` 3-Attr on `gt_hul_sessions`** with `<60s` catalog extensibility across all `2,659` SKUs. Best suited for high-volume GT Kirana audits where 3-Attribute Share-of-Shelf is the primary contract metric.
2. **MVP Arm B (Primary Production Champion): `960px Soft-NMS ONNX + Stage-2A CLM Fast Gate (59% zero-generative-VLM) + Stage-2B Contrastive CLEF (Gemini 3.5 Flash-Lite)`**
   * **Role in MVP**: Resolves ~60% of easy crops via Arm A at 0 VLM cost, and routes the ~40% hard sister variants and promo packs to `Gemini 3.5 Flash-Lite` with bottom-30% Zoom-Strip Contact Sheets. Delivers **`91.6%` 3-Attr / `82.6%+` 5-Attr on `hul_golden`**, **`88.3%` on `hul_promos`**, **`80.8%` on brand-new zero-shot SKUs**, and **`1.88%` SOS error on wild `gtmerch_golden` shelves** with zero retraining.
3. **MVP Arm C (Maximum-Precision Specialist for Priority MT/GT Accounts): `960px Soft-NMS ONNX + Stage-2A CLM Fast Gate + Stage-2B Vertex LoRA SFT Gemini 3.5 Flash-Lite`**
   * **Role in MVP**: Swaps the zero-shot Stage-2B VLM call for the fine-tuned Vertex AI LoRA SFT endpoint (`#6726259894726295552`), lifting fine-grained **L6 Variant accuracy to `95.9%` and L8 E2E accuracy to `93.5%`**, while retaining the `hul_gallery` retrieval path for newly launched SKUs.


## 7. Next-Wave Architectural Improvements & Experiments (`EXP-28` to `EXP-32`)

Error trace analysis on `1013-025001-l8_hul_ultimate-gemini-3.5-flash-lite-priority-hul_variants` ($N=1,000$) revealed that **`gemini-embedding-2` Top-5 Recall (`Recall@5`) is already `93.3%`**: among the `182` wrong predictions, the true ground-truth SKU is already present in the Top-5 shortlist in **`115 / 182` (`63.2%`)** of errors (and Brand is already right in **`153 / 182 = 84.1%`**).

To capture that `11.5 pp` headroom (`81.8% -> 93.3%`) and lift wild-shelf Stage-1 detection (`61.6%-62.3%` $F_2$), we have designed five targeted experiments:

1. **`EXP-28` (`clef_contrastive_metadata`): Contrastive Metadata-Enriched CLEF Contact Sheets**
   * *Gap Addressed*: Previously, `CLEF_ZOOM_SINGLE_PROMPT` sent the 6-panel visual contact sheet (`Q` + `Panels 1..5`) with **zero text labels** for `Panels 1..5`, forcing `Gemini 3.5 Flash-Lite` to read tiny `256px` reference thumbnails to guess whether Panel 1 was `SPF 30` or `SPF 50`, or `Tub` vs `Jar`.
   * *Design*: Inject each candidate's catalog metadata (`Panel k: <Brand> | <Base_Variant> | <Pack_Type>`) plus **automated pairwise contrastive diff tokens** (such as `diff: [Spf_30] vs [Spf_50]` or `diff: [Jar] vs [Tub]`) directly into the prompt so the VLM focuses 100% of its OCR attention on Query `Q`.
2. **`EXP-29` (`clm_multiangle_gallery`): Multi-Angle Shelf Medoid Gallery Augmentation**
   * *Gap Addressed*: In the `6.7%` of test crops where the true SKU missed the Top-5 shortlist (`67 / 182` errors), the query was a tilted or side-angled shelf crop while `rpc.json["gallery"]` only held front studio views.
   * *Design*: Augment `rpc.json["gallery"]` with up to `2` real-shelf k-means medoid vectors per SKU mined from `splits["train"]` (`0` overlap with `val`/`test`).
3. **`EXP-30` (`metrics.py` Canonical Equivalence): Bilingual & Container-Suffix Twin Normalization**
   * *Gap Addressed*: Legacy folder splits created exact physical twin classes (`Pepsodent__Clove_And_Salt` vs `Pepsodent__Square_Lavang_And_Salt_Toothpaste` where *Lavang* = Hindi for *Clove*, `Kissan__Mixed_Fruit` vs `Kissan__Mixed_Fruit_Tub` when `packaging_type == Tub`, `Bournvita__Strength` vs `Bournvita__Strength_Pouch`).
   * *Design*: Normalize bilingual reverse-panel twins and redundant container suffixes when `packaging_type` matches, reporting both raw and canonicalized accuracy.
4. **`EXP-31` (`distilled_yolo_960_split_nms`): High-Res Tile Container-Splitting NMS on Wild Shelves**
   * *Gap Addressed*: On `ms_json_golden` and `gtmerch_golden`, coarse `640x640` full-frame boxes occasionally merge two adjacent sachets/pouches into a single box and suppress the accurate high-res `960px` sub-window detections (`IoU >= 0.28`).
   * *Design*: When $\ge 2$ mutually non-overlapping (`IoU < 0.25`) high-res tile boxes fall inside (`containment >= 0.80`) a single coarse full-frame box, replace the merged coarse box with the individual high-res tile boxes.
5. **`EXP-32` (`dino_clm_clef_cascade`): 3-Tier Local + Cloud Hybrid Cascade**
   * *Gap Addressed*: Every crop currently invokes a network RPC to `gemini-embedding-2`.
   * *Design*: Use local `DINOv2/v3` SFT strictly as a **Tier-0 Ultra-High-Margin Gate (`conf >= 0.85, margin >= 0.55`)** for known head SKUs (`>98.5%` precision), and route all other crops (long-tail, low-margin, new launches, and open-set competitors) to `gemini-embedding-2` CLM + `Gemini 3.5 Flash-Lite` CLEF.
