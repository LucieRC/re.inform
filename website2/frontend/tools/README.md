# Frontend tools

## Extract personalities SVGs

If you have a single file containing multiple `<svg>...</svg>` blocks (like [website2/frontend/personalities_svg_code.xml](../personalities_svg_code.xml)), you can split it into individual icon files under `assets/personalities/`.

Run:

```bash
python3 website2/frontend/tools/extract_personality_svgs.py \
  website2/frontend/personalities_svg_code.xml \
  website2/frontend/assets/personalities \
  --limit 10
```

This writes `p01.svg`, `p02.svg`, … in the same order as the `<svg>` blocks appear in the input file.
