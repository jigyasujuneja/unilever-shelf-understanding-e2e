# Ground Truth Contract

This is the specification the annotation vendor delivers against and the specification the
benchmark reads. Both sides work from this file.

Ground truth **does not exist yet** for this project. Until it does, the suite runs, produces
predictions, latency and cost, and reports accuracy as `null` with
`accuracy_status = "PLACEHOLDER_AWAITING_GROUND_TRUTH"` — see
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) section 1. The day a file lands, every run you
have ever done can be scored against it for free.

Source of truth:
[`data/ground_truth.py`](../src/shelf_benchmark/data/ground_truth.py),
[`models.py`](../src/shelf_benchmark/models.py) (`ImageGroundTruth`, `GroundTruthProductItem`) and
`GroundTruthConfig` / `GroundTruthSchemaMapping` in
[`config.py`](../src/shelf_benchmark/config.py).

---

## 1. What is being annotated

One photograph of a retail shelf. The annotator marks every **front-facing product unit** and
labels it.

A *facing* is one product presented to the shopper at the front of a shelf column. If four
identical bottles are stacked front-to-back in the same column, that is **one facing**, not four:
the shopper sees one. The unit at the front is the facing; the ones behind it are back-row units.

Annotate the back-row units too, flagged with `back_row: true`. The suite drops them by default
(`ground_truth.exclude_back_row_items = true`) so that detection recall is measured against what
is actually visible, but having them recorded means the choice stays reversible and
depth-deduplication logic can be evaluated later.

---

## 2. Canonical JSON schema

The suite-native format. Everything else is adapted onto this.

### Top level

| Key | Required | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `gt_version` | recommended | string | Revision label for this annotation set. Overrides the configured `gt_version`. |
| `bbox_format` | recommended | string | One of the five values in section 4. Checked against the config; a disagreement is a hard error. |
| `image_width` / `image_height` | conditional | int | Fallback pixel dimensions, required only for pixel bbox formats when an entry omits its own. |
| `images` | required | object | Map of image key to image entry. May be omitted, in which case the root object itself is treated as the map. |

### Image entry

| Key | Required | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `image_id` | required | string | How the benchmark refers to this image. See section 3. |
| `items` | required | array | One object per annotated unit, including back-row units. |
| `total_main_shelf_facings` | recommended | int | Front facings on this shelf. Becomes `ground_truth_count`, the denominator of recall. Falls back to the number of non-back-row items. |
| `expected_brands` | recommended | array of string | Distinct brands present. Denominator of `brand_set_recall`. Falls back to the distinct brands of the kept items. |
| `image_width` / `image_height` | conditional | int | Pixel dimensions of this photo. Required for pixel bbox formats. |

### Item

| Key | Required | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `item_id` | recommended | int | Stable identifier within the image. Defaults to the 1-based position in `items`. |
| `brand` | **required** | string | Brand exactly as printed on the pack. Scored by `brand_classification_accuracy`. |
| `product_name` | **required** | string | Full product display name. Scored by `product_classification_accuracy`. |
| `bbox_2d` | **required** | array of 4 numbers | Bounding box in the declared `bbox_format`. |
| `sku_id` | recommended | string | Canonical SKU or EAN. Scored by `sku_matching_accuracy` when the prediction also has one. |
| `shelf_row` | recommended | string | `top`, `middle`, `bottom`, or a bay label. Defaults to `middle`. |
| `back_row` | recommended | bool | `true` for a unit stacked behind a front facing. Excluded from scoring by default. |
| `occluded` | recommended | bool | `true` when substantially hidden by another product. Kept and scored. |
| `category` | optional | string | Taxonomy dimension 1. |
| `subcategory` | optional | string | Taxonomy dimension 2. |
| `variant` | optional | string | Taxonomy dimension 4. |
| `packaging_type` | optional | string | Taxonomy dimension 5, e.g. `tube`, `bottle`, `sachet`. |
| `pack_type` | optional | string | Taxonomy dimension 6, `Single` or `Multiple`. |
| `size` | optional | string | Taxonomy dimension 7, the printed net weight or volume. |

Booleans may be delivered as real JSON booleans or as any of the strings
`true`, `1`, `yes`, `y`, `t` (case-insensitive). Anything else, including absence, means `false`.

### Complete worked example

```json
{
  "gt_version": "v1.0-pilot-mumbai",
  "bbox_format": "ymin_xmin_ymax_xmax_1000",
  "images": {
    "gs://unilever-shelf-understanding-shelf-images/store-4471/aisle-3.png": {
      "image_id": "gs://unilever-shelf-understanding-shelf-images/store-4471/aisle-3.png",
      "image_width": 4032,
      "image_height": 3024,
      "total_main_shelf_facings": 3,
      "expected_brands": ["Pond's", "Himalaya", "Lakme"],
      "items": [
        {
          "item_id": 1,
          "brand": "Pond's",
          "product_name": "Pond's Bright Beauty Face Wash",
          "sku_id": "HUL-PONDS-BB-100G",
          "category": "Skin Cleansing",
          "subcategory": "Face Wash",
          "variant": "Bright Beauty",
          "packaging_type": "tube",
          "pack_type": "Single",
          "size": "100g",
          "bbox_2d": [100, 100, 300, 200],
          "shelf_row": "top",
          "back_row": false,
          "occluded": false
        },
        {
          "item_id": 2,
          "brand": "Himalaya",
          "product_name": "Himalaya Neem Face Wash",
          "sku_id": "HIM-NEEM-100G",
          "category": "Skin Cleansing",
          "subcategory": "Face Wash",
          "variant": "Purifying Neem",
          "packaging_type": "tube",
          "pack_type": "Single",
          "size": "100g",
          "bbox_2d": [100, 220, 300, 320],
          "shelf_row": "top",
          "back_row": false,
          "occluded": true
        },
        {
          "item_id": 3,
          "brand": "Lakme",
          "product_name": "Lakme Blush and Glow Face Wash",
          "sku_id": "LKM-BG-100G",
          "category": "Skin Cleansing",
          "subcategory": "Face Wash",
          "variant": "Blush and Glow",
          "packaging_type": "tube",
          "pack_type": "Single",
          "size": "100g",
          "bbox_2d": [400, 100, 600, 200],
          "shelf_row": "middle",
          "back_row": false,
          "occluded": false
        },
        {
          "item_id": 4,
          "brand": "Pond's",
          "product_name": "Pond's Bright Beauty Face Wash",
          "sku_id": "HUL-PONDS-BB-100G",
          "bbox_2d": [100, 100, 300, 200],
          "shelf_row": "top",
          "back_row": true,
          "occluded": true
        }
      ]
    }
  }
}
```

Loading that file reports:

```
JSONGroundTruthProvider: 1 image(s), 3 item(s), 1 back-row item(s) excluded, gt_version=v1.0-pilot-mumbai
```

Three items kept, item 4 dropped as a back-row unit. `total_main_shelf_facings` is 3, so
`detection_recall` will be measured against 3.

### The copy that ships with the repo

The same schema, filled in and ready to load, is
[`configs/sample_ground_truth.json`](../configs/sample_ground_truth.json): one image, three
front-row facings with all seven taxonomy dimensions populated, and one `back_row: true` unit so
you can watch the exclusion happen. Use it as the reference when briefing a vendor, and as the
smoke test for your own plumbing:

```bash
.venv/bin/shelf-benchmark validate-gt \
  --gt-provider json \
  --ground-truth-uri configs/sample_ground_truth.json
```

```
JSONGroundTruthProvider: 1 image(s), 3 item(s), 1 back-row item(s) excluded, gt_version=sample-v1
```

Every knob mentioned in this document is also present, commented, in
[`configs/default_config.yaml`](../configs/default_config.yaml) under `ground_truth:` and
`evaluation:`, which is the discoverable list: provider type, source URI, `gt_version`, `strict`,
`exclude_back_row_items`, the full `schema_mapping`, and all five `bbox_format` values.

### Generating a starter file

`shelf_benchmark.testing.write_sample_ground_truth_file()` writes a valid file in exactly this
schema, keyed on whichever image you name. Use it when you want a throwaway file in a test:

```python
from shelf_benchmark.testing import write_sample_ground_truth_file

path = write_sample_ground_truth_file("annotations_smoke_test.json")
print(path)
```

---

## 3. Image keys and how lookup works

The `image_id` is the join key between annotations and benchmark runs. Getting it wrong is the
single most common integration failure, so the provider indexes each entry under several aliases:

* the key used in the `images` map,
* the entry's `image_id`,
* the basename of each of those (the part after the last `/`).

Lookup then tries, in order: the run's `shelf_image_uri`, an explicit `ground_truth_id`, and the
basename of the `shelf_image_uri`.

That means both of these resolve to the same entry:

```
gs://unilever-shelf-understanding-shelf-images/store-4471/aisle-3.png
aisle-3.png
```

**Recommendation for the vendor:** use the full `gs://` URI of the shelf image as `image_id`.
It is unambiguous, and basename matching still works if a consumer only has the filename.

> [!WARNING]
> Basename matching means two shelf images called `aisle-3.png` in different store folders will
> collide. If your dataset has repeated basenames, the full URI must be the key, and consumers
> must look up by full URI.

---

## 4. Bounding box formats

Internally the suite works in `[ymin, xmin, ymax, xmax]` normalized to `0..1000`. Annotation
vendors overwhelmingly ship COCO `[x, y, width, height]` in absolute pixels. Guessing between
them produces detection numbers that are plausible, stable, and completely meaningless — so the
format must be **declared**, never inferred.

Declare it in `ground_truth.schema_mapping.bbox_format`, on the CLI with `--bbox-format`, or via
`sdk.connect_ground_truth(bbox_format=<value>)`. Optionally also declare it as a top-level
`bbox_format` in the JSON file itself; if the file and the config disagree the load fails rather
than picking one.

### All five supported values

`convert_bbox(raw, fmt, image_width, image_height)` implements every conversion.

| `bbox_format` | Input layout | Units | Needs image size? | Conversion to `[ymin, xmin, ymax, xmax]` on 0-1000 |
| :--- | :--- | :--- | :--- | :--- |
| `ymin_xmin_ymax_xmax_1000` | `[ymin, xmin, ymax, xmax]` | already 0-1000 | no | identity |
| `coco_xywh_px` | `[x, y, width, height]` | absolute pixels | **yes** | `ymin = y/H*1000`, `xmin = x/W*1000`, `ymax = (y+height)/H*1000`, `xmax = (x+width)/W*1000` |
| `xyxy_px` | `[x1, y1, x2, y2]` | absolute pixels | **yes** | `ymin = y1/H*1000`, `xmin = x1/W*1000`, `ymax = y2/H*1000`, `xmax = x2/W*1000` |
| `xyxy_norm` | `[x1, y1, x2, y2]` | fractions of 0-1 | no | `ymin = y1*1000`, `xmin = x1*1000`, `ymax = y2*1000`, `xmax = x2*1000` |
| `yxyx_norm` | `[y1, x1, y2, x2]` | fractions of 0-1 | no | multiply all four by 1000 |

Results are rounded to integers and clamped to `[0, 1000]`.

Pixel formats need `image_width` and `image_height`. They are read, in order, from the item's
image entry (`image_width_field` / `image_height_field`, default `image_width` / `image_height`),
then from the file's top level, then from
`schema_mapping.default_image_width` / `default_image_height`. If none are available,
`convert_bbox` raises `GroundTruthError` rather than assuming a size.

### Worked numeric example for `coco_xywh_px`

A 4032x3024 photograph. The product occupies pixels `x = 1008` to `1512` and `y = 756` to `2268`.
COCO records that as `[x, y, width, height] = [1008, 756, 504, 1512]`.

```
W, H = 4032, 3024
x, y, w, h = 1008, 756, 504, 1512

xmin =  x        / W * 1000 = 1008 / 4032 * 1000 = 250
ymin =  y        / H * 1000 =  756 / 3024 * 1000 = 250
xmax = (x + w)   / W * 1000 = 1512 / 4032 * 1000 = 375
ymax = (y + h)   / H * 1000 = 2268 / 3024 * 1000 = 750

bbox_2d = [ymin, xmin, ymax, xmax] = [250, 250, 750, 375]
```

Note the reordering as well as the rescaling. A box that is narrow and tall in pixels
(`504` wide by `1512` high) is narrow and tall in the normalized space too (`125` wide by `500`
high), because both axes are scaled independently by their own dimension. Aspect ratio is **not**
preserved by this transform, and it does not need to be: IoU is computed in the same space on
both sides.

The exact same physical box in all five formats:

| `bbox_format` | Raw value in the file | Result |
| :--- | :--- | :--- |
| `ymin_xmin_ymax_xmax_1000` | `[250, 250, 750, 375]` | `[250, 250, 750, 375]` |
| `coco_xywh_px` | `[1008, 756, 504, 1512]` | `[250, 250, 750, 375]` |
| `xyxy_px` | `[1008, 756, 1512, 2268]` | `[250, 250, 750, 375]` |
| `xyxy_norm` | `[0.25, 0.25, 0.375, 0.75]` | `[250, 250, 750, 375]` |
| `yxyx_norm` | `[0.25, 0.25, 0.75, 0.375]` | `[250, 250, 750, 375]` |

Check it yourself:

```python
from shelf_benchmark.data.ground_truth import convert_bbox

print(convert_bbox([1008, 756, 504, 1512], "coco_xywh_px", 4032, 3024))
print(convert_bbox([1008, 756, 1512, 2268], "xyxy_px", 4032, 3024))
print(convert_bbox([0.25, 0.25, 0.375, 0.75], "xyxy_norm"))
print(convert_bbox([0.25, 0.25, 0.75, 0.375], "yxyx_norm"))
print(convert_bbox([250, 250, 750, 375], "ymin_xmin_ymax_xmax_1000"))
```

All five print `[250, 250, 750, 375]`.

> [!CAUTION]
> `xyxy_norm` and `yxyx_norm` are trivially confusable and produce *valid-looking* boxes when
> swapped. Feeding `[0.25, 0.25, 0.375, 0.75]` through `yxyx_norm` yields
> `[250, 250, 375, 750]` — a wide flat box where a tall narrow one belongs. Nothing errors. Run
> `validate-gt` (section 7) and look at the printed boxes.

---

## 5. Supported providers

Set `ground_truth.provider_type` (or `--gt-provider`, or `connect_ground_truth(provider_type=<value>)`).

| `provider_type` | Source | Shape expected | Notes |
| :--- | :--- | :--- | :--- |
| `none` | nothing | — | Default. Produces `PLACEHOLDER_AWAITING_GROUND_TRUTH`. |
| `json` | local path or `gs://` URI | Either the canonical object-of-images shown in section 2, or a JSON array of image entries each carrying `image_key_field` | Reads the top-level `gt_version` and `bbox_format` when present |
| `jsonl` | local path or `gs://` URI | One image entry per line | Same code path as `json`. See the gotcha below. |
| `csv` | local path or `gs://` URI | One row per **item**, grouped by `image_key_field` | `bbox_field` holds a JSON array as a string. Missing mapped columns are a hard error listing the columns that are present. |
| `coco` | local path or `gs://` URI | Standard COCO: `images` + `annotations` + `categories` | Geometry needs no configuration: boxes are COCO `xywh` pixels and dimensions come from the `images` array |
| `bigquery` | `project.dataset.table`, or a full `SELECT` | One row per item | Requires credentials. `source_uri` starting with `SELECT` is run as-is, otherwise it becomes `SELECT * FROM \`<table>\``. |

> [!NOTE]
> A JSONL file containing exactly **one** line is also valid JSON, so it is parsed as a single
> object rather than as a one-element list — and a single object is interpreted as a map of
> images, which almost certainly yields zero parsed images. If you are testing with one image,
> use the canonical `json` object form instead.

### JSON array form

When the root is an array, each element is an image entry and must carry the image key under
`image_key_field`:

```json
[
  {
    "image_id": "store-4471/aisle-3.png",
    "image_width": 4032,
    "image_height": 3024,
    "items": [{"item_id": 1, "brand": "Pond's", "product_name": "Pond's Bright Beauty Face Wash",
               "bbox_2d": [100, 100, 300, 200], "shelf_row": "top"}]
  }
]
```

### CSV form

One row per annotated unit. The `bbox_2d` column holds a JSON array, so it needs quoting:

```csv
image_id,item_id,brand,product_name,sku_id,bbox_2d,shelf_row,back_row,occluded
store-4471/aisle-3.png,1,Pond's,Pond's Bright Beauty Face Wash,HUL-PONDS-BB-100G,"[100, 100, 300, 200]",top,false,false
store-4471/aisle-3.png,2,Himalaya,Himalaya Neem Face Wash,HIM-NEEM-100G,"[100, 220, 300, 320]",top,false,true
```

For pixel bbox formats, add `image_width` and `image_height` columns; the first non-empty value
per image group is used.

### COCO form

```json
{
  "images": [{"id": 1, "file_name": "aisle-3.png", "width": 4032, "height": 3024}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1008, 756, 504, 1512],
     "attributes": {"brand": "Pond's", "product_name": "Pond's Bright Beauty Face Wash",
                    "sku_id": "HUL-PONDS-BB-100G", "shelf_row": "top",
                    "back_row": false, "occluded": false}}
  ],
  "categories": [{"id": 1, "name": "Pond's Bright Beauty Face Wash"}]
}
```

COCO has nowhere standard to put brand, SKU or shelf row, so the provider reads them from
`annotations[].attributes` (which CVAT, Label Studio and Roboflow all emit). When
`attributes.brand` or `attributes.product_name` is absent it falls back to the category name, so a
COCO file whose categories *are* the products still works. `iscrowd` is treated as `occluded`.

The image key is `coco_url`, else `file_name`, else the numeric `id`.

---

## 6. Mapping an arbitrary vendor schema

You will not get the canonical field names. `GroundTruthSchemaMapping` renames every field without
touching any code or transforming the vendor's file.

| Mapping attribute | Default field name | Points at |
| :--- | :--- | :--- |
| `image_key_field` | `image_id` | Image key |
| `items_list_field` | `items` | The per-image array of units |
| `item_id_field` | `item_id` | Item identifier |
| `brand_field` | `brand` | Brand |
| `product_name_field` | `product_name` | Product name |
| `sku_id_field` | `sku_id` | SKU / EAN |
| `bbox_field` | `bbox_2d` | Bounding box (falls back to `bbox` if the mapped name is absent) |
| `shelf_row_field` | `shelf_row` | Shelf row |
| `back_row_field` | `back_row` | Back-row flag |
| `occluded_field` | `occluded` | Occlusion flag |
| `category_field` | `category` | Taxonomy dimension 1 |
| `subcategory_field` | `subcategory` | Taxonomy dimension 2 |
| `variant_field` | `variant` | Taxonomy dimension 4 |
| `packaging_type_field` | `packaging_type` | Taxonomy dimension 5 |
| `pack_type_field` | `pack_type` | Taxonomy dimension 6 |
| `size_field` | `size` | Taxonomy dimension 7 |
| `image_width_field` | `image_width` | Per-image pixel width |
| `image_height_field` | `image_height` | Per-image pixel height |
| `default_image_width` | `None` | Fallback pixel width |
| `default_image_height` | `None` | Fallback pixel height |
| `bbox_format` | `ymin_xmin_ymax_xmax_1000` | Section 4 |

### Concrete example

The vendor delivers JSONL that looks like this:

```json
{"photo_ref": "store-4471/aisle-3.png", "px_w": 4032, "px_h": 3024,
 "detections": [
   {"ann_id": "1", "manufacturer": "Pond's", "sku_description": "Pond's Bright Beauty Face Wash",
    "ean": "HUL-PONDS-BB-100G", "box": [1008, 756, 504, 1512], "bay_level": "top",
    "is_behind": "false", "is_occluded": "no"},
   {"ann_id": "2", "manufacturer": "Himalaya", "sku_description": "Himalaya Neem Face Wash",
    "ean": "HIM-NEEM-100G", "box": [1512, 756, 504, 1512], "bay_level": "top",
    "is_behind": "true", "is_occluded": "yes"}]}
{"photo_ref": "store-4471/aisle-4.png", "px_w": 4032, "px_h": 3024,
 "detections": [
   {"ann_id": "1", "manufacturer": "Lakme", "sku_description": "Lakme Blush and Glow Face Wash",
    "ean": "LKM-BG-100G", "box": [504, 1512, 504, 1512], "bay_level": "middle",
    "is_behind": "false", "is_occluded": "no"}]}
```

Nothing matches the defaults, the boxes are COCO pixels, and the booleans are strings. In Python:

```python
from shelf_benchmark import ShelfBenchmarkSDK

sdk = ShelfBenchmarkSDK(config_path="configs/default_config.yaml")
stats = sdk.connect_ground_truth(
    provider_type="jsonl",
    source_uri="vendor_v1.jsonl",
    bbox_format="coco_xywh_px",
    gt_version="v1.0-pilot-mumbai",
    schema_mapping={
        "image_key_field": "photo_ref",
        "items_list_field": "detections",
        "item_id_field": "ann_id",
        "brand_field": "manufacturer",
        "product_name_field": "sku_description",
        "sku_id_field": "ean",
        "bbox_field": "box",
        "shelf_row_field": "bay_level",
        "back_row_field": "is_behind",
        "occluded_field": "is_occluded",
        "image_width_field": "px_w",
        "image_height_field": "px_h",
    },
)
print(stats["summary"])
```

Or in YAML, so the CLI, the Web UI and every SDK run agree:

```yaml
ground_truth:
  provider_type: "jsonl"
  source_uri: "gs://unilever-shelf-understanding-shelf-images/gt/vendor_v1.jsonl"
  gt_version: "v1.0-pilot-mumbai"
  strict: true
  exclude_back_row_items: true
  schema_mapping:
    image_key_field: "photo_ref"
    items_list_field: "detections"
    item_id_field: "ann_id"
    brand_field: "manufacturer"
    product_name_field: "sku_description"
    sku_id_field: "ean"
    bbox_field: "box"
    shelf_row_field: "bay_level"
    back_row_field: "is_behind"
    occluded_field: "is_occluded"
    image_width_field: "px_w"
    image_height_field: "px_h"
    bbox_format: "coco_xywh_px"
```

Loading the JSONL above gives:

```
JSONGroundTruthProvider: 2 image(s), 2 item(s), 1 back-row item(s) excluded, gt_version=v1.0-pilot-mumbai
```

and the Pond's box comes out as `[250, 250, 750, 375]`.

> [!TIP]
> `connect_ground_truth` validates every `schema_mapping` key against the real attribute names and
> raises with the full list of valid keys if you typo one. A typo used to be ignored, leaving the
> default field name in place and silently producing empty ground truth.

---

## 7. Verifying: the `validate-gt` loop

Do this **before** spending money on a scored run.

```bash
.venv/bin/shelf-benchmark validate-gt \
  --offline \
  --gt-provider json \
  --ground-truth-uri annotations_v1.json \
  --gt-version v1.0-pilot-mumbai \
  --sample 3
```

```
OK: JSONGroundTruthProvider: 1 image(s), 3 item(s), 1 back-row item(s) excluded, gt_version=v1.0-pilot-mumbai
    bbox_format : ymin_xmin_ymax_xmax_1000
    gt_version  : v1.0-pilot-mumbai
    images      : 1
    items       : 3
    excluded_back_row: 1

First 1 parsed entries (verify the boxes look sane):
  key='aisle-3.png' facings=3 version=v1.0-pilot-mumbai
      [1] Pond's | Pond's Bright Beauty Face Wash | bbox=[100, 100, 300, 200]
      [2] Himalaya | Himalaya Neem Face Wash | bbox=[100, 220, 300, 320]
      [3] Lakme | Lakme Blush and Glow Face Wash | bbox=[400, 100, 600, 200]

Boxes are shown in suite-native [ymin, xmin, ymax, xmax] on a 0-1000 scale. If these do not
correspond to where the products actually are, your --bbox-format is wrong.
```

Work down this checklist:

1. **`images` is the number of photographs you expect.** Zero means the key field is wrong; see
   section 8.
2. **`items` is roughly the number of facings you expect.** An order of magnitude out means the
   item list field points at the wrong array.
3. **`excluded_back_row` is plausible.** Zero on a dataset that should have depth stacking means
   `back_row_field` is not being read, or the vendor sent a value the boolean parser does not
   recognize.
4. **`gt_version` is the one you meant.** If the file declares its own it wins over the config,
   and `validate-gt` says so explicitly.
5. **The printed boxes make sense.** This is the step that catches a wrong `bbox_format`. A
   product on the left of the shelf should have a small `xmin`; a product on the top shelf should
   have a small `ymin`. A box where `ymax - ymin` is much larger than `xmax - xmin` is a tall
   narrow product. If a wide flat box comes out where a tall bottle should be, your axis order is
   swapped.
6. **Repeat with `--bbox-format` set to what you think the vendor used**, and compare. The wrong
   format rarely errors — it just moves the boxes.

Then dry-run the join on real predictions:

```bash
.venv/bin/shelf-benchmark score \
  --predictions reports/predictions.json \
  --ground-truth-uri annotations_v1.json \
  --gt-version v1.0-pilot-mumbai
```

If the keys do not line up this raises `ScoringError` immediately rather than emitting zeros.

---

## 8. `gt_version` and why it matters

```yaml
ground_truth:
  gt_version: "v1.0-pilot-mumbai"
```

`gt_version` is stamped onto:

* `AccuracyMetrics.gt_version` in `benchmark_summary.json`,
* the `gt_version` column of every row in `row_level_report.csv`,
* every OpenTelemetry span.

Annotations get corrected. A vendor fixes twenty mislabelled variants, or the team decides that
promotional multipacks count as one facing rather than three. Every accuracy number computed
before that change is now incomparable with every number computed after it — and nothing about the
model changed.

Because re-scoring is free, the discipline is cheap:

```bash
.venv/bin/shelf-benchmark score --predictions reports/predictions.json \
  --ground-truth-uri annotations_v1.json --gt-version v1 --output-dir reports/v1
.venv/bin/shelf-benchmark score --predictions reports/predictions.json \
  --ground-truth-uri annotations_v2.json --gt-version v2 --output-dir reports/v2
```

Same predictions, two ground-truth revisions, two labelled result sets. When someone asks why
brand accuracy jumped six points, you can answer.

A file-level `gt_version` overrides the configured one, so the vendor can make their revision
authoritative. Use a label that names the dataset and the revision, not just a number:
`v1.0-pilot-mumbai`, `v1.1-pilot-mumbai-variant-fixes`.

---

## 9. `back_row` and `occluded`

These two flags are handled very differently and the difference is deliberate.

| Flag | Meaning | Default handling | Controlled by |
| :--- | :--- | :--- | :--- |
| `back_row` | Unit stacked behind a front facing in the same shelf column | **Excluded** from the item list, from `total_main_shelf_facings` fallback, from `expected_brands` fallback, and from every metric | `ground_truth.exclude_back_row_items` (default `true`) |
| `occluded` | Unit substantially hidden by a neighbouring product, but still the front-most unit in its column | **Kept and scored normally** | nothing — always scored |

`back_row` is excluded because the benchmark measures front facings. A model given a photograph
cannot see the fourth bottle in a stack, so counting it as a false negative would penalize the
model for the laws of optics. The count reported to the business is a facings count, which is what
share-of-shelf is computed from.

`occluded` is kept because a partially hidden product is genuinely on the shelf and a good model
should find it. It is recorded so that you can slice the results — an approach that scores well
overall but collapses on occluded facings is a real and useful finding — but it never changes the
denominator.

To include back-row units (for example when evaluating depth-deduplication logic itself):

```yaml
ground_truth:
  exclude_back_row_items: false
```

Every provider reports how many it dropped, in `load_stats["excluded_back_row"]` and in
`describe()`. If that number is `0` on a dataset you know has depth stacking, the flag is not
being read.

---

## 10. Failing loudly

`ground_truth.strict` defaults to `true`. Under it, these all raise `GroundTruthError` at load
time rather than degrading to "no ground truth available":

* The source URI cannot be read.
* The file is empty, or is neither valid JSON nor valid JSONL.
* Zero images parsed.
* An image entry has no item list under `items_list_field`.
* An item has no bounding box under `bbox_field` (or `bbox`).
* A CSV is missing one of the mapped `image_key_field`, `brand_field` or `bbox_field` columns.
* `bbox_format` is unknown, or is a pixel format with no image dimensions available.
* The file declares a `bbox_format` that disagrees with the configured one.

There is also `ground_truth.min_matched_image_ratio`: when set above `0` and `strict` is on, the
run raises if fewer than that fraction of benchmarked images resolve to a ground-truth entry. Use
it in CI to catch a partial join.

"Ground truth is not configured yet" and "ground truth is configured and broken" produce very
different reports and must never look identical. Setting `strict: false` downgrades all of the
above to warnings; only do that when you genuinely want a partially-annotated dataset to run.

---

## 11. Troubleshooting

### Zero images parsed

```
GroundTruthError: JSONGroundTruthProvider loaded 0 ground-truth images from 'annotations_v1.json'.
This usually means a wrong URI or a wrong `image_key_field` in the schema mapping.
```

or, more helpfully, when the array form is used:

```
GroundTruthError: Ground-truth entry is missing the image key field 'image_id'.
Available fields: ['items', 'picture']
```

Causes, in order of likelihood:

1. `image_key_field` names a field the vendor does not use. The error lists the fields that *are*
   present — map to one of those.
2. The file is a JSON **array** but `image_key_field` is not set, so no entry has a key.
3. The file is a single-line JSONL, which parses as one JSON object and is then read as a map of
   images. Use the canonical object form.
4. The top-level object is a map of images but the values are not objects — for example the root
   is an `images` key holding a list instead of a map.

Run `validate-gt` and read the `images:` line. It answers this in one second.

### Key mismatch: parsed fine, matched nothing

```
ScoringError: None of the 3 run(s) in 'reports/predictions.json' matched a ground-truth entry.
This usually means the ground-truth key field does not correspond to the image URIs used at run
time. Check `image_key_field` in your schema mapping, and compare the keys in your annotations
against the `shelf_image_uri` values in the predictions file. Scoring aborted rather than
reporting zeros.
```

The annotations loaded, but nothing joined. Compare the two sides directly:

```bash
.venv/bin/python -c "
import json
runs = json.load(open('reports/predictions.json'))['runs']
print('prediction image URIs:', sorted({r['shelf_image_uri'] for r in runs}))
"

.venv/bin/shelf-benchmark validate-gt --offline \
  --gt-provider json --ground-truth-uri annotations_v1.json --sample 10
```

The `key=` values printed by `validate-gt` must overlap the prediction URIs, either exactly or by
basename. Common causes: the vendor keyed on a database row id instead of the filename; the
benchmark ran against a local path while the annotations use `gs://`; a trailing slash or a
different bucket name.

### Plausible-but-wrong numbers: the bbox format is wrong

The worst failure, because nothing errors. Symptoms:

* `detection_precision` and `detection_recall` are low but non-zero, around 0.1 to 0.4.
* `mean_iou_matched` is mediocre where it should be high.
* `count_accuracy` is `1.0` — the model found the right number of things.
* `brand_classification_accuracy` is much lower than a human reading the crops would expect,
  because predictions are pairing with the wrong ground-truth items or not pairing at all.
* Raising `iou_threshold` collapses recall to zero far faster than it should.

That combination — right count, wrong geometry — is almost always a coordinate convention
mismatch, not a model problem. Check, in this order:

1. Run `validate-gt` and look at the printed boxes against the actual photograph. A product at
   the left edge must have a small `xmin`.
2. Axis order. `xyxy_norm` versus `yxyx_norm`, and `coco_xywh_px` versus `xyxy_px`, are the two
   confusable pairs. Swapping either produces well-formed boxes in the wrong places.
3. `width, height` versus `x2, y2`. If the vendor sends `xyxy` but you declared `coco_xywh_px`,
   every box becomes enormous, because `x2` is read as a width and added to `x`.
4. Image dimensions. If `image_width` and `image_height` are transposed, every box is skewed by
   the aspect ratio. On a 4:3 photo that is a 1.33x error in one axis — enough to wreck IoU,
   small enough to look like a mediocre model.
5. Origin. This suite assumes the origin is top-left, y increasing downwards, which is the
   convention for COCO, PIL and Gemini. A bottom-left origin mirrors every box vertically.

### Everything is `null` and `accuracy_status` says `PLACEHOLDER`

Ground truth was never connected for that image. Either `provider_type` is still `none`, or the
provider loaded but the lookup missed. `BaseGroundTruthProvider.unmatched_lookups` lists the image
URIs that were requested and not found — check it before assuming the file is empty.

### `bbox_format` mismatch between file and config

```
GroundTruthError: Ground-truth file declares bbox_format='ymin_xmin_ymax_xmax_1000' but the config
says 'coco_xywh_px'. Fix the config rather than letting boxes be misinterpreted.
```

Working as intended. The file's declaration is the vendor's statement of fact; fix the config.

### CSV: missing mapped column

```
GroundTruthError: Ground-truth source 'annotations_v1.csv' is missing mapped column(s)
{'brand_field': 'manufacturer'}. Columns present: ['image_id', 'item_id', 'brand',
'product_name', 'sku_id', 'bbox_2d', 'shelf_row', 'back_row'].
Update GroundTruthSchemaMapping so the join cannot silently return nothing.
```

The error prints the real header row. Map to one of those names.

---

## 12. Delivery checklist for the annotation vendor

- [ ] One entry per shelf photograph, keyed by the photograph's full `gs://` URI.
- [ ] Every front-facing product unit annotated, including own-brand and competitor products.
- [ ] Depth-stacked units annotated with `back_row: true`.
- [ ] Partially hidden but front-most units annotated with `occluded: true` and `back_row: false`.
- [ ] `brand` transcribed exactly as printed, including punctuation.
- [ ] `product_name` is the full display name, not an abbreviation or an internal code.
- [ ] `sku_id` populated wherever the SKU is determinable.
- [ ] `bbox_format` declared once at the top of the file, and consistent throughout.
- [ ] `image_width` and `image_height` present on every entry if any pixel format is used.
- [ ] `total_main_shelf_facings` present and equal to the count of non-`back_row` items.
- [ ] `expected_brands` present and equal to the distinct brands of the non-`back_row` items.
- [ ] `gt_version` set to a label naming the dataset and the revision.
- [ ] The file passes `shelf-benchmark validate-gt` with the expected image and item counts, and
      spot-checked boxes that land on the right products.
