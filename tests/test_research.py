"""
Research mode retrieval test — no model required.
Tests ResearchRetriever against all three sources with a real query.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from Hydrusopt import ResearchRetriever, ArtifactRenderer, _html_module

QUERIES = [
    "sleep and memory consolidation",
    "transformer attention mechanism",
]

retriever_full = ResearchRetriever(
    sources=["arxiv", "semanticscholar", "pubmed"],
    max_results_per_source=2,
)

artifact_renderer = ArtifactRenderer(output_dir="artifacts")

all_html_rows = []

for query in QUERIES:
    print(f"\n{'='*60}")
    print(f"  Query: {query}")
    print(f"{'='*60}")

    results = retriever_full.fetch(query)

    if not results:
        print("  [!] No results returned — check network access.")
        continue

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] ({r['source']}) {r['title']}")
        print(f"       {r['snippet'][:120]}...")
        if r.get("url"):
            print(f"       {r['url']}")

    # Format context block as the model would see it
    context = retriever_full.format_context(results)
    print(f"\n  --- Context block (first 400 chars) ---")
    print(f"  {context[:400]}")

    # Build HTML rows for the artifact
    for r in results:
        escaped_title   = _html_module.escape(r["title"])
        escaped_snippet = _html_module.escape(r["snippet"][:200])
        url = r.get("url", "")
        link = f'<a href="{_html_module.escape(url)}" target="_blank">{escaped_title}</a>' if url else escaped_title
        all_html_rows.append(
            f"<tr><td><span class='src'>{_html_module.escape(r['source'])}</span></td>"
            f"<td>{link}</td>"
            f"<td>{escaped_snippet}</td></tr>"
        )

# Render results as an HTML artifact
if all_html_rows:
    rows_html = "\n".join(all_html_rows)
    html_body = f"""
<style>
  body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f6f8fa; }}
  h2 {{ color: #0d1117; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; vertical-align: top; }}
  th {{ background: #161b22; color: #c9d1d9; font-size: 12px; }}
  tr:hover td {{ background: #f0f6ff; }}
  .src {{ background: #1f3a5f; color: #79c0ff; border-radius: 3px;
          padding: 1px 6px; font-size: 11px; white-space: nowrap; }}
  a {{ color: #0969da; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
<h2>HydrusOPT Research Mode — Retrieval Test</h2>
<p style="color:#555;font-size:13px">Queries: {_html_module.escape(' | '.join(QUERIES))}</p>
<table>
  <tr><th>Source</th><th>Title</th><th>Snippet</th></tr>
  {rows_html}
</table>
"""
    path = artifact_renderer.render_html(html_body, title="research_results.html")
    abs_path = os.path.abspath(path)
    print(f"\n  [ARTIFACT] Results rendered → {abs_path}")
    import webbrowser
    webbrowser.open(f"file:///{abs_path}")

print("\nDone.")
