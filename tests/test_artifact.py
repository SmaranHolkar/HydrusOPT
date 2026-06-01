"""
Quick test for ArtifactRenderer — no model required.
Renders three artifact types and opens them in the default browser.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from Hydrusopt import ArtifactRenderer

renderer = ArtifactRenderer(output_dir="artifacts")

# 1 — Python code artifact
py_code = '''\
def greet(name: str) -> str:
    return f"Hello, {name}!"

for name in ["HydrusOPT", "World", "Skills"]:
    print(greet(name))
'''
p1 = renderer.render_code(py_code, lang="python", title="greet.py")
print(f"[1] Python artifact → {os.path.abspath(p1)}")

# 2 — HTML preview artifact
html_body = '''\
<h1 style="font-family:sans-serif;color:#0d6efd">HydrusOPT Artifact Preview</h1>
<p style="font-family:sans-serif;color:#333">
  This is a rendered <strong>HTML artifact</strong> from the ArtifactRenderer.
</p>
<ul style="font-family:sans-serif;color:#555">
  <li>Skills (tool-calling) ✓</li>
  <li>Artifact rendering ✓</li>
  <li>Chat loop ✓</li>
</ul>
'''
p2 = renderer.render_html(html_body, title="preview.html")
print(f"[2] HTML  artifact → {os.path.abspath(p2)}")

# 3 — detect_and_render from raw model-like output
model_output = '''\
Here is a simple counter component:

```html
<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;padding:20px">
  <h2>Counter</h2>
  <p id="val">0</p>
  <button onclick="document.getElementById('val').textContent=+document.getElementById('val').textContent+1">+1</button>
  <button onclick="document.getElementById('val').textContent=+document.getElementById('val').textContent-1">-1</button>
</body>
</html>
```
'''
paths = renderer.detect_and_render(model_output)
for i, p in enumerate(paths, 3):
    print(f"[{i}] Auto-detected artifact → {os.path.abspath(p)}")

import webbrowser
for p in [p1, p2] + paths:
    webbrowser.open(f"file:///{os.path.abspath(p)}")

print("\nDone — check your browser.")
