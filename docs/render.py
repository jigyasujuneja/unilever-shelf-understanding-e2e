"""Render docs/diagrams/*.mmd to docs/images/*.svg and *.png with Mermaid in headless Chromium.

    make docs        # or: uv run --no-project --with playwright python docs/render.py
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
PAGE = """<!doctype html><html><body style="margin:0;background:#fff">
<div id="out" style="display:inline-block;padding:24px"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:false, theme:"neutral",
  fontFamily:"Arial, Helvetica, sans-serif", flowchart:{htmlLabels:true, curve:"basis", wrappingWidth:420},
  sequence:{mirrorActors:false}});</script></body></html>"""


async def main() -> None:
    (HERE / "images").mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=2, viewport={"width": 1600, "height": 1000})
        await page.set_content(PAGE)
        await page.wait_for_function("window.mermaid !== undefined")
        for src in sorted((HERE / "diagrams").glob("*.mmd")):
            size = await page.evaluate(
                "async ([id, code]) => { const {svg} = await mermaid.render(id, code);"
                " const out = document.getElementById('out'); out.innerHTML = svg;"
                " const el = out.querySelector('svg'); el.style.maxWidth = 'none';"
                " const vb = el.viewBox.baseVal; el.setAttribute('width', vb.width);"
                " el.setAttribute('height', vb.height);"
                " return [Math.ceil(vb.width) + 48, Math.ceil(vb.height) + 48]; }",
                [src.stem, src.read_text()])
            await page.set_viewport_size({"width": size[0], "height": size[1]})
            (HERE / "images" / f"{src.stem}.svg").write_text(
                await page.locator("#out").inner_html())
            await page.locator("#out").screenshot(path=str(HERE / "images" / f"{src.stem}.png"))
            print(json.dumps({"rendered": src.stem, "size": size}))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
