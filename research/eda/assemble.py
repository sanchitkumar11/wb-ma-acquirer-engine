"""Inject the computed JSON into the page template.

Usage: python3 -I assemble.py <template.html> <data.json> <out.html> [d3 src override]
"""
import sys

tpl, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
html = open(tpl, encoding="utf-8").read()
payload = open(data, encoding="utf-8").read().replace("</", "<\\/")
assert html.count("__DATA__") == 1
html = html.replace("__DATA__", payload)
if len(sys.argv) > 4:  # local preview only: swap the CDN script for a local copy
    html = html.replace("https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js", sys.argv[4])
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, f"{len(html) / 1024:.0f} KB")
