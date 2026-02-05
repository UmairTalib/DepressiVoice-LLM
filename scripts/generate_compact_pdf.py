#!/usr/bin/env python3
"""
Generate compact PDF from condensed markdown report
"""

import markdown
import re
from pathlib import Path
from weasyprint import HTML

def embed_images(md_content, md_file_path):
    """Convert relative image paths to absolute paths"""
    md_dir = Path(md_file_path).parent
    
    def replace_image(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        
        if rel_path.startswith('http'):
            return match.group(0)
        
        img_path = (md_dir / rel_path).resolve()
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            return match.group(0)
        
        return f'![{alt_text}](file://{img_path})'
    
    pattern = r'!\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, replace_image, md_content)

def generate_compact_pdf(md_file, pdf_file):
    """Convert markdown to compact PDF"""
    
    print(f"Reading markdown: {md_file}")
    with open(md_file, 'r') as f:
        md_content = f.read()
    
    print("Resolving image paths...")
    md_content = embed_images(md_content, md_file)
    
    print("Converting to HTML...")
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.toc'
        ]
    )
    
    # Compact CSS for better page utilization
    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Phase 1 Technical Report</title>
    <style>
        @page {{
            size: Letter;
            margin: 0.6in 0.75in;
            @bottom-center {{
                content: "Phase 1 Technical Report - Page " counter(page);
                font-size: 8pt;
                color: #666;
            }}
        }}
        
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            line-height: 1.5;
            color: #333;
            font-size: 9.5pt;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin: 16pt 0 12pt 0;
            font-size: 16pt;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #95a5a6;
            padding-bottom: 4px;
            margin: 14pt 0 8pt 0;
            page-break-after: avoid;
            font-size: 13pt;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin: 10pt 0 6pt 0;
            page-break-after: avoid;
            font-size: 11pt;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 1px 4px;
            border-radius: 2px;
            font-family: 'Courier New', monospace;
            font-size: 8.5pt;
        }}
        
        pre {{
            background-color: #f4f4f4;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            page-break-inside: avoid;
            font-size: 7.5pt;
            line-height: 1.3;
            margin: 8pt 0;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 8pt 0;
            page-break-inside: avoid;
            font-size: 8.5pt;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 6px 4px;
            text-align: left;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        img {{
            max-width: 95%;
            height: auto;
            margin: 10pt auto;
            page-break-inside: avoid;
            display: block;
        }}
        
        ul, ol {{
            margin: 6pt 0;
            padding-left: 25px;
        }}
        
        li {{
            margin: 3pt 0;
        }}
        
        p {{
            margin: 6pt 0;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 12pt 0;
        }}
        
        /* Better page breaks */
        h1, h2, h3 {{
            page-break-after: avoid;
        }}
        
        h1:first-of-type {{
            margin-top: 0;
        }}
        
        p {{ orphans: 2; widows: 2; }}
        
        figure {{ page-break-inside: avoid; }}
        
        /* Compact lists */
        ul li, ol li {{ line-height: 1.4; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    print(f"Generating PDF: {pdf_file}")
    HTML(string=styled_html, base_url=str(Path(md_file).parent)).write_pdf(pdf_file)
    
    print(f"✅ PDF generated successfully!")
    print(f"📄 Output: {pdf_file}")
    
    size_kb = Path(pdf_file).stat().st_size / 1024
    print(f"📊 Size: {size_kb:.1f} KB")

if __name__ == "__main__":
    md_file = "/Users/umairtalib/DepressiVoice-LLM/docs/reports/Phase1_Technical_Report_Condensed.md"
    pdf_file = "/Users/umairtalib/DepressiVoice-LLM/docs/reports/Phase1_Technical_Report.pdf"
    
    generate_compact_pdf(md_file, pdf_file)
