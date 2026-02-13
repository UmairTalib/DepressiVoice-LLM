#!/usr/bin/env python3
"""
Export Phase 4 Technical Report to HTML and PDF.
"""

import subprocess
from pathlib import Path
import os
import sys

def export_report():
    root = Path(__file__).parent.parent.parent
    md_file = root / "docs/reports/PHASE4_TECHNICAL_REPORT.md"
    html_file = root / "docs/reports/PHASE4_TECHNICAL_REPORT.html"
    pdf_file = root / "docs/reports/PHASE4_TECHNICAL_REPORT.pdf"
    
    if not md_file.exists():
        print(f"Error: {md_file} not found")
        return False

    print(f"Converting {md_file.name} to HTML...")
    
    # 1. MD to HTML
    try:
        import markdown
        with open(md_file, 'r') as f:
            text = f.read()
        
        # Extensions for tables and math (if needed)
        html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
        
        css_url = "https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css"
        full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Phase 4 Technical Report - Session-Level Temporal Aggregation</title>
<link rel="stylesheet" href="{css_url}">
<style>
    body {{ 
        box-sizing: border-box; 
        min-width: 200px; 
        max-width: 980px; 
        margin: 0 auto; 
        padding: 45px; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; 
    }}
    @media (max-width: 767px) {{ 
        body {{ padding: 15px; }} 
    }}
    img {{ 
        max-width: 80%; 
        height: auto; 
        display: block; 
        margin: 20px auto; 
        border: 1px solid #ddd; 
        border-radius: 4px; 
        padding: 5px; 
    }}
    h1 {{ 
        border-bottom: 2px solid #eaecef; 
        padding-bottom: 0.3em; 
        color: #0366d6;
    }}
    h2 {{ 
        border-bottom: 1px solid #eaecef; 
        padding-bottom: 0.3em; 
        margin-top: 24px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }}
    th, td {{
        border: 1px solid #dfe2e5;
        padding: 8px 13px;
        text-align: left;
    }}
    th {{
        background-color: #f6f8fa;
        font-weight: 600;
    }}
    code {{
        background-color: #f6f8fa;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 85%;
    }}
    pre {{
        background-color: #f6f8fa;
        padding: 16px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
        border-radius: 6px;
    }}
    blockquote {{
        padding: 0 1em;
        color: #6a737d;
        border-left: 0.25em solid #dfe2e5;
        margin: 0 0 16px 0;
    }}
    .markdown-body {{ 
        font-size: 16px; 
        line-height: 1.6; 
        word-wrap: break-word; 
    }}
</style>
</head>
<body class="markdown-body">
{html_content}
</body>
</html>"""
        
        with open(html_file, 'w') as f:
            f.write(full_html)
        print(f"✓ HTML generated: {html_file.name}")
        
    except Exception as e:
        print(f"✗ HTML generation failed: {e}")
        return False

    # 2. HTML to PDF
    print(f"Converting {html_file.name} to PDF...")
    try:
        # Import the logic from html_to_pdf.py
        sys.path.append(str(root / "scripts"))
        from html_to_pdf import html_to_pdf
        
        success = html_to_pdf(str(html_file), str(pdf_file))
        if success:
            print(f"✓ PDF generated: {pdf_file.name}")
            print(f"\n📄 Report Files:")
            print(f"   Markdown: {md_file}")
            print(f"   HTML:     {html_file}")
            print(f"   PDF:      {pdf_file}")
            return True
        else:
            print("✗ PDF generation failed via Chrome headless.")
            return False
            
    except Exception as e:
        print(f"✗ PDF generation error: {e}")
        print(f"   HTML file available at: {html_file}")
        return False

if __name__ == "__main__":
    success = export_report()
    sys.exit(0 if success else 1)
