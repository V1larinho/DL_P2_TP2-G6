import pathlib
import re
import bs4
import markdown
from playwright.sync_api import sync_playwright
import pypdf

# Define directories and file paths
doc_dir = pathlib.Path(__file__).resolve().parent.parent / "docs"
md_path = doc_dir / "P2-D1.md"
pdf_path = doc_dir / "P2-D1.pdf"
html_path = doc_dir / "P2-D1.html"
img_path = doc_dir / "mockup_anotacao.png"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Convert markdown to raw HTML
raw_html = markdown.markdown(text, extensions=["tables", "fenced_code"])
soup = bs4.BeautifulSoup(raw_html, "html.parser")

# Split elements into Page 1 and Page 2
p1_elements = []
p2_elements = []
current_page = 1

for el in list(soup.children):
    if not getattr(el, "name", None):
        continue
    # Switch to Page 2 at Section 4
    if el.name == "h2" and "4. Aplicação" in el.text:
        current_page = 2
    
    if current_page == 1:
        p1_elements.append(el)
    else:
        p2_elements.append(el)

# --- Executive Academic Header for Page 1 ---
header_html = """
<header class="doc-header">
  <div class="header-top-bar">
    <div class="header-academic">
      <span class="degrees">MEI / MEB / MECD / MMC</span>
      <span class="sep">·</span>
      <span class="uc">DEEP LEARNING APLICADO (2026/2027)</span>
    </div>
    <div class="header-status">
      <span class="badge-deliverable">PROPOSTA DE PROJETO · P2-D1</span>
    </div>
  </div>
  <div class="header-main">
    <h1 class="doc-title">Active Learning para defeitos em pontes de betão: quantos rótulos são mesmo precisos?</h1>
    <div class="header-meta">
      <span class="meta-pill"><strong>Tema C (AL 1):</strong> Família C — Poucos Rótulos</span>
      <span class="meta-pill"><strong>Turno:</strong> TP2</span>
      <span class="meta-pill"><strong>Grupo:</strong> 6</span>
      <span class="meta-pill"><strong>Data:</strong> 29 de Outubro de 2026</span>
    </div>
  </div>
</header>
"""

# Filter & enrich Page 1 elements
filtered_p1 = []
skip_next_hr = False

for el in p1_elements:
    if el.name == "h1":
        continue
    if el.name == "p" and "Tema C (AL 1)" in el.text:
        continue
    if el.name == "p" and "Título de trabalho:" in el.text:
        skip_next_hr = True
        continue
    if el.name == "hr" and skip_next_hr:
        skip_next_hr = False
        continue
    filtered_p1.append(el)

# Format Members Table (Table 1)
for el in filtered_p1:
    if el.name == "table":
        el["class"] = el.get("class", []) + ["table-members"]
        # Enhance role cells
        for tr in el.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 4:
                # Member name
                tds[0]["class"] = tds[0].get("class", []) + ["col-member"]
                tds[1]["class"] = tds[1].get("class", []) + ["col-num"]
                tds[2]["class"] = tds[2].get("class", []) + ["col-role"]
                tds[3]["class"] = tds[3].get("class", []) + ["col-task"]
                
                role_text = tds[2].decode_contents()
                # Format A1..A5 badges
                role_text = re.sub(
                    r"<strong>(A[1-5])</strong>\s*[—–-]\s*",
                    r'<span class="badge-role">\1</span> ',
                    role_text,
                )
                role_text = re.sub(
                    r"\b(A[1-5])\b\s*[—–-]\s*",
                    r'<span class="badge-role">\1</span> ',
                    role_text,
                )
                role_text = re.sub(
                    r"<strong>\(porta-voz\)</strong>|\(<strong>porta-voz</strong>\)|\(porta-voz\)",
                    r'<span class="badge-spokesperson">porta-voz</span>',
                    role_text,
                )
                tds[2].clear()
                tds[2].append(bs4.BeautifulSoup(role_text, "html.parser"))
        break

# Format Table 2.1 and Parameters Table
table_count = 0
for el in filtered_p1:
    if el.name == "table":
        table_count += 1
        if table_count == 2:
            el["class"] = el.get("class", []) + ["table-problem-analysis"]
            for tr in el.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    tds[0]["class"] = tds[0].get("class", []) + ["col-pergunta"]
                    tds[1]["class"] = tds[1].get("class", []) + ["col-resposta"]
        elif table_count == 3:
            el["class"] = el.get("class", []) + ["table-sim-params"]
            for tr in el.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    tds[0]["class"] = tds[0].get("class", []) + ["col-param"]
                    tds[1]["class"] = tds[1].get("class", []) + ["col-param-val"]

# Enhance section headers and elements in P1
for el in filtered_p1:
    if el.name == "h2":
        el["class"] = el.get("class", []) + ["sec-heading"]
        m = re.match(r"(\d+)\.\s*(.*)", el.text.strip())
        if m:
            sec_num, sec_title = m.groups()
            el.clear()
            el.append(
                bs4.BeautifulSoup(
                    f'<span class="sec-num">§{sec_num}</span><span class="sec-title">{sec_title}</span>',
                    "html.parser",
                )
            )
    elif el.name == "h3":
        el["class"] = el.get("class", []) + ["subsec-heading"]
    elif el.name == "blockquote":
        el["class"] = el.get("class", []) + ["callout-box"]
        # Add callout header
        inner_content = el.decode_contents()
        new_callout = f"""
        <div class="callout-header"><span class="callout-tag">Nota Metodológica & Auditoria dos Dados</span></div>
        <div class="callout-body">{inner_content}</div>
        """
        el.clear()
        el.append(bs4.BeautifulSoup(new_callout, "html.parser"))
    elif el.name == "p" and "Hipóteses." in el.text:
        # Style hypotheses as structured items
        content = el.decode_contents()
        content = re.sub(
            r"<strong>Hipóteses\.</strong>\s*",
            r'<div class="hyp-title"><strong>Hipóteses de Investigação:</strong></div>',
            content,
        )
        content = re.sub(
            r"<strong>H1</strong>\s*[—–-]\s*",
            r'<div class="hyp-card"><span class="badge-hyp">H1</span> ',
            content,
        )
        content = re.sub(
            r"<strong>H2</strong>\s*[—–-]\s*",
            r'</div><div class="hyp-card"><span class="badge-hyp">H2</span> ',
            content,
        )
        content = re.sub(
            r"<strong>H3</strong>\s*[—–-]\s*",
            r'</div><div class="hyp-card"><span class="badge-hyp">H3</span> ',
            content,
        )
        content = re.sub(
            r"<strong>H4</strong>\s*\(exploratória\)\s*[—–-]\s*",
            r'</div><div class="hyp-card"><span class="badge-hyp">H4 (exploratória)</span> ',
            content,
        )
        content += "</div>"
        el.clear()
        el.append(bs4.BeautifulSoup(content, "html.parser"))

# --- Process Page 2 elements ---
img_uri = img_path.as_uri()
filtered_p2 = []

for el in p2_elements:
    if el.name == "h2":
        el["class"] = el.get("class", []) + ["sec-heading"]
        m = re.match(r"(\d+)\.\s*(.*)", el.text.strip())
        if m:
            sec_num, sec_title = m.groups()
            if sec_num == "5":
                sec_title = "Plano de Trabalho, Cronograma e Estimativa Computacional"
            el.clear()
            el.append(
                bs4.BeautifulSoup(
                    f'<span class="sec-num">§{sec_num}</span><span class="sec-title">{sec_title}</span>',
                    "html.parser",
                )
            )
        filtered_p2.append(el)
    elif el.name == "p" and el.find("img"):
        # Wrap mockup image in an executive framed figure
        fig_html = f"""
        <figure class="mockup-figure">
          <div class="mockup-frame">
            <img src="{img_uri}" alt="Assistente de anotação — Interface Gradio (Mockup)">
          </div>
          <figcaption><strong>Figura 1</strong> — Assistente de anotação interativo (Gradio) com re-treino em tempo real e visualização da composição de classes rotuladas.</figcaption>
        </figure>
        """
        filtered_p2.append(bs4.BeautifulSoup(fig_html, "html.parser"))
    elif el.name == "table":
        # Table 5.1: Schedule or Table 5.2: Computational
        if not any("table-schedule" in x.get("class", []) for x in filtered_p2 if getattr(x, "name", None) == "table"):
            el["class"] = el.get("class", []) + ["table-schedule"]
            for tr in el.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 3:
                    tds[0]["class"] = tds[0].get("class", []) + ["col-semana"]
                    tds[1]["class"] = tds[1].get("class", []) + ["col-marco"]
                    tds[2]["class"] = tds[2].get("class", []) + ["col-resp"]
                    
                    resp_html = tds[2].decode_contents()
                    resp_html = re.sub(r"\b(A[1-5])\b", r'<span class="badge-role-mini">\1</span>', resp_html)
                    tds[2].clear()
                    tds[2].append(bs4.BeautifulSoup(resp_html, "html.parser"))
            filtered_p2.append(el)
        else:
            el["class"] = el.get("class", []) + ["table-computational"]
            for tr in el.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 3:
                    tds[0]["class"] = tds[0].get("class", []) + ["col-fase"]
                    tds[1]["class"] = tds[1].get("class", []) + ["col-calc"]
                    tds[2]["class"] = tds[2].get("class", []) + ["col-medir"]
            filtered_p2.append(el)
    else:
        filtered_p2.append(el)

body_p1 = "".join(str(el) for el in filtered_p1)
body_p2 = "".join(str(el) for el in filtered_p2)

# Full HTML template
full_html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="utf-8">
<title>P2-D1 — Proposta do Projeto (Tema C: Active Learning)</title>
<style>
  @page {{
    size: A4;
    margin: 0.58cm 0.72cm 0.80cm 0.72cm;
  }}

  :root {{
    --primary: #1e3a8a;          /* Deep Navy */
    --primary-dark: #0f172a;     /* Slate 900 */
    --primary-accent: #2563eb;   /* Vibrant Blue */
    --primary-light: #eff6ff;    /* Soft Blue Tint */
    --primary-border: #bfdbfe;   /* Blue Border */
    --slate-text: #1e293b;       /* Slate 800 */
    --slate-muted: #475569;      /* Slate 600 */
    --slate-light: #f8fafc;      /* Slate 50 */
    --slate-alt: #f1f5f9;        /* Slate 100 */
    --border-color: #cbd5e1;     /* Slate 300 */
    --border-subtle: #e2e8f0;    /* Slate 200 */
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 6.5pt;
    line-height: 1.18;
    color: var(--slate-text);
    margin: 0;
    padding: 0;
    background: #ffffff;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}

  .page-container {{
    width: 100%;
    position: relative;
  }}

  .page-break {{
    page-break-before: always;
    break-before: page;
  }}

  /* --- Header Styling --- */
  .doc-header {{
    border-bottom: 2px solid var(--primary-accent);
    padding-bottom: 2px;
    margin-bottom: 2.5px;
  }}

  .header-top-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 0.5px solid var(--border-subtle);
    padding-bottom: 1.5px;
    margin-bottom: 1.5px;
  }}

  .header-academic {{
    font-size: 6.0pt;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: var(--primary);
  }}

  .header-academic .sep {{
    color: var(--border-color);
    margin: 0 4px;
  }}

  .badge-deliverable {{
    background: var(--primary);
    color: #ffffff;
    font-size: 5.5pt;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 1.2px 5px;
    border-radius: 2.5px;
    text-transform: uppercase;
  }}

  .doc-title {{
    font-size: 10.2pt;
    font-weight: 800;
    color: var(--primary-dark);
    line-height: 1.14;
    margin: 1.5px 0 1.5px 0;
    letter-spacing: -0.2px;
  }}

  .header-meta {{
    font-size: 5.8pt;
    color: var(--slate-muted);
    display: flex;
    align-items: center;
    gap: 4px;
  }}

  .meta-pill {{
    background: var(--slate-light);
    border: 0.5px solid var(--border-subtle);
    padding: 0.5px 4px;
    border-radius: 2px;
  }}

  /* --- Section Headings --- */
  .sec-heading {{
    font-size: 7.6pt;
    font-weight: 700;
    color: var(--primary);
    margin: 3.0px 0 1.2px 0;
    padding-bottom: 1px;
    border-bottom: 0.8px solid var(--border-color);
    display: flex;
    align-items: center;
    letter-spacing: 0.2px;
    text-transform: uppercase;
  }}

  .sec-num {{
    background: var(--primary);
    color: #ffffff;
    font-size: 6.0pt;
    font-weight: 700;
    padding: 0.5px 3.5px;
    border-radius: 2px;
    margin-right: 4px;
    display: inline-block;
  }}

  .subsec-heading {{
    font-size: 6.6pt;
    font-weight: 700;
    color: var(--primary-dark);
    margin: 2px 0 1px 0;
  }}

  /* --- Paragraphs and Lists --- */
  p {{
    margin: 0 0 1.6px 0;
    text-align: justify;
    text-justify: inter-word;
  }}

  ul, ol {{
    margin: 0 0 1.8px 0;
    padding-left: 10px;
  }}

  li {{
    margin-bottom: 0.5px;
    text-align: justify;
  }}

  /* --- Tables Styling --- */
  table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 5.8pt;
    margin: 1.8px 0;
    line-height: 1.10;
    border: 0.5px solid var(--border-color);
    border-radius: 3px;
    overflow: hidden;
  }}

  th {{
    background-color: var(--primary);
    color: #ffffff;
    font-weight: 700;
    font-size: 5.8pt;
    letter-spacing: 0.2px;
    padding: 1.6px 3.5px;
    text-align: left;
    border: none;
    border-bottom: 0.5px solid var(--primary-dark);
  }}

  td {{
    border: none;
    border-bottom: 0.5px solid var(--border-subtle);
    border-right: 0.5px solid var(--border-subtle);
    padding: 1.2px 3.2px;
    vertical-align: top;
    color: var(--slate-text);
  }}

  td:last-child {{
    border-right: none;
  }}

  tr:last-child td {{
    border-bottom: none;
  }}

  tr:nth-child(even) td {{
    background-color: var(--slate-light);
  }}

  tr:nth-child(odd) td {{
    background-color: #ffffff;
  }}

  /* --- Column Widths & Layout --- */
  .table-members .col-member {{ width: 21%; font-weight: 600; color: var(--primary-dark); }}
  .table-members .col-num {{ width: 11%; font-family: Consolas, monospace; color: var(--slate-muted); }}
  .table-members .col-role {{ width: 30%; }}
  .table-members .col-task {{ width: 38%; }}

  .table-problem-analysis .col-pergunta {{
    width: 26%;
    font-weight: 700;
    color: var(--primary);
    background-color: var(--slate-alt);
    border-right: 0.8px solid var(--border-color);
  }}
  .table-problem-analysis .col-resposta {{
    width: 74%;
  }}

  .table-sim-params .col-param {{
    width: 25%;
    font-weight: 700;
    color: var(--primary-dark);
    background-color: var(--slate-alt);
    border-right: 0.8px solid var(--border-color);
  }}
  .table-sim-params .col-param-val {{
    width: 75%;
  }}

  .table-schedule .col-semana {{ width: 14%; font-weight: 600; background-color: var(--slate-alt); }}
  .table-schedule .col-marco {{ width: 56%; }}
  .table-schedule .col-resp {{ width: 30%; }}

  .table-computational .col-fase {{ width: 20%; font-weight: 600; background-color: var(--slate-alt); }}
  .table-computational .col-calc {{ width: 52%; }}
  .table-computational .col-medir {{ width: 28%; }}

  /* --- Badges --- */
  .badge-role {{
    display: inline-block;
    background: #e0e7ff;
    color: #1e3a8a;
    font-weight: 700;
    font-size: 5.5pt;
    padding: 0.4px 3.5px;
    border-radius: 2.5px;
    border: 0.5px solid #c7d2fe;
    margin-right: 2px;
  }}

  .badge-role-mini {{
    display: inline-block;
    background: #e0e7ff;
    color: #1e3a8a;
    font-weight: 700;
    font-size: 5.2pt;
    padding: 0.3px 3px;
    border-radius: 2px;
    border: 0.5px solid #c7d2fe;
  }}

  .badge-spokesperson {{
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    font-weight: 700;
    font-size: 5.0pt;
    padding: 0.3px 3px;
    border-radius: 2px;
    border: 0.5px solid #bfdbfe;
    text-transform: uppercase;
    margin-left: 2px;
  }}

  .badge-hyp {{
    display: inline-block;
    background: var(--primary);
    color: #ffffff;
    font-weight: 700;
    font-size: 5.5pt;
    padding: 0.4px 3.5px;
    border-radius: 2px;
    margin-right: 3px;
    vertical-align: middle;
  }}

  .hyp-title {{
    font-weight: 700;
    color: var(--primary-dark);
    margin-top: 1px;
    margin-bottom: 1px;
  }}

  .hyp-card {{
    background: var(--slate-light);
    border: 0.5px solid var(--border-subtle);
    border-left: 2px solid var(--primary-accent);
    padding: 1.0px 4px;
    margin: 1px 0;
    border-radius: 0 2px 2px 0;
    text-align: justify;
  }}

  /* --- Callout Box --- */
  .callout-box {{
    border-left: 3px solid var(--primary-accent);
    background-color: var(--primary-light);
    border-top: 0.5px solid var(--primary-border);
    border-right: 0.5px solid var(--primary-border);
    border-bottom: 0.5px solid var(--primary-border);
    border-radius: 0 3px 3px 0;
    margin: 2px 0;
    padding: 2px 5px;
    font-size: 6.1pt;
    line-height: 1.15;
    color: var(--slate-text);
  }}

  .callout-header {{
    margin-bottom: 1px;
  }}

  .callout-tag {{
    font-size: 5.2pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    background: var(--primary);
    color: #ffffff;
    padding: 0.3px 3.5px;
    border-radius: 2px;
  }}

  .callout-body p {{
    margin: 0 0 1px 0;
  }}

  .callout-body p:last-child {{
    margin-bottom: 0;
  }}

  /* --- Code --- */
  code {{
    background-color: var(--slate-alt);
    padding: 0 2px;
    border-radius: 2px;
    font-family: Consolas, "JetBrains Mono", monospace;
    font-size: 5.9pt;
    color: var(--primary-dark);
    border: 0.5px solid var(--border-subtle);
  }}

  /* --- Page 2 Specifics & Mockup Figure --- */
  #page-2 {{
    font-size: 7.0pt;
    line-height: 1.25;
  }}

  #page-2 p {{
    margin: 0 0 2.5px 0;
  }}

  #page-2 .sec-heading {{
    font-size: 7.8pt;
    margin: 5px 0 2px 0;
  }}

  #page-2 table {{
    font-size: 6.3pt;
    line-height: 1.16;
    margin: 3.5px 0;
  }}

  #page-2 td {{
    padding: 2.0px 4px;
  }}

  #page-2 th {{
    font-size: 6.3pt;
    padding: 2.2px 4px;
  }}

  .mockup-figure {{
    margin: 4px auto 3px auto;
    text-align: center;
  }}

  .mockup-frame {{
    display: inline-block;
    background: #ffffff;
    padding: 2.5px;
    border: 0.8px solid var(--border-color);
    border-radius: 4px;
    box-shadow: 0 1.5px 5px rgba(0, 0, 0, 0.08);
  }}

  .mockup-frame img {{
    display: block;
    max-height: 250px;
    max-width: 96%;
    margin: 0 auto;
    border-radius: 2px;
  }}

  figcaption {{
    font-size: 6.3pt;
    color: var(--slate-muted);
    margin-top: 2.5px;
    font-style: italic;
  }}

  figcaption strong {{
    font-style: normal;
    color: var(--primary);
  }}
</style>
</head>
<body>

<div class="page-container" id="page-1">
{header_html}
{body_p1}
</div>

<div class="page-break"></div>

<div class="page-container" id="page-2">
{body_p2}
</div>

</body>
</html>
"""

# Write out the HTML
with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

# Generate PDF with Playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(html_path.as_uri())
    page.wait_for_timeout(1000)
    
    # Save preview screenshots of Page 1 and Page 2 for visual QA
    page.locator("#page-1").screenshot(path=str(doc_dir / "preview_p1.png"))
    page.locator("#page-2").screenshot(path=str(doc_dir / "preview_p2.png"))

    footer_tmpl = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 6.2pt; width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 0.72cm; color: #64748b; border-top: 0.5px solid #cbd5e1; padding-top: 2.5px;">
      <span><strong>MEI/MEB/MECD/MMC</strong> · Deep Learning Aplicado (2026/2027) · TP2 — Grupo 6 (Tema C: Active Learning)</span>
      <span>Página <span class="pageNumber"></span> de <span class="totalPages"></span></span>
    </div>
    """

    page.pdf(
        path=str(pdf_path),
        format="A4",
        margin={"top": "0.58cm", "bottom": "0.80cm", "left": "0.72cm", "right": "0.72cm"},
        display_header_footer=True,
        header_template="<div></div>",
        footer_template=footer_tmpl,
        print_background=True,
    )
    browser.close()

reader = pypdf.PdfReader(str(pdf_path))
num_pages = len(reader.pages)
print(f"Resultado final: {num_pages} página(s) gerada(s).")
