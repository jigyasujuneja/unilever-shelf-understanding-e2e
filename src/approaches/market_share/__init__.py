"""Market Share & Product Counting approaches (``use_case = "market_share"``).

Subdirectories mirror the UI task tabs:
- ``detection/``: locate every product facing on SKU-110K shelf photos.
- ``classification/``: identify a product photo from the closed FMCG catalog.
- ``retrieval/``: identify a product crop against reference gallery photos.
- ``end_to_end/``: full shelf/checkout pipelines (composed via ``compose()`` or single-call via
  ``detect_and_identify()``).
"""
