#!/usr/bin/env python3
"""
Generate PDF from comprehensive markdown report with proper image handling
"""

import markdown
import re
import base64
from pathlib import Path
from weasyprint import HTML

def embed_images(md_content, md_file_path):
    """Convert relative image paths to absolute paths or embed as base64"""
    
    md_dir = Path(md_file_path).parent
    
    def replace_image(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        
        # Construct absolute path
        if rel_path.startswith('http'):
            # External URL, keep as is
            return match.group(0)
        
        # Handle relative paths
        img_path = (md_dir / rel_path).resolve()
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            return match.group(0)
        
        # Return absolute file:// URL
        return f'![{alt_text}](file://{img_path})'
    
    # Replace all markdown image references
    pattern = r'!\[(.*?)\]\((.*?)\)'
    return re.sub(pattern, replace_image, md_content)

def generate_pdf(md_file, pdf_file):
    """Convert markdown to PDF with properly embedded images"""
    
    print(f"Reading markdown: {md_file}")
    with open(md_file, 'r') as f:
        md_content = f.read()
    
    # Fix image paths
    print("Resolving image paths...")
    md_content = embed_images(md_content, md_file)
    
    # Convert to HTML with extensions
    print("Converting to HTML...")
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.toc'
        ]
    )
    
    # Add professional CSS
    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Phase 1 Comprehensive Technical Report</title>
    <style>
        @page {{
            size: Letter;
            margin: 0.75in;
            @bottom-center {{
                content: "Phase 1 Technical Report - Page " counter(page);
                font-size: 9pt;
                color: #666;
            }}
        }}
        
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 10pt;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            page-break-before: always;
            margin-top: 0;
            font-size: 18pt;
        }}
        
        h1:first-of-type {{
            page-break-before: avoid;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 24pt;
            page-break-after: avoid;
            font-size: 14pt;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin-top: 16pt;
            page-break-after: avoid;
            font-size: 12pt;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }}
        
        pre {{
            background-color: #f4f4f4;
            padding: 12px;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
            font-size: 8pt;
            line-height: 1.4;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12pt 0;
            page-break-inside: avoid;
            font-size: 9pt;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 6px;
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
            max-width: 100%;
            height: auto;
            margin: 12pt 0;
            page-break-inside: avoid;
            display: block;
        }}
        
        ul, ol {{
            margin: 8pt 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 4pt 0;
        }}
        
        p {{
            margin: 8pt 0;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 20pt 0;
        }}
        
        /* Prevent orphans and widows */
        p {{
            orphans: 3;
            widows: 3;
        }}
        
        /* Better page breaks */
        h1, h2, h3, h4 {{
            page-break-after: avoid;
        }}
        
        /* Keep figures with their captions */
        figure {{
            page-break-inside: avoid;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Save HTML for debugging
    html_file = str(pdf_file).replace('.pdf', '_debug.html')
    with open(html_file, 'w') as f:
        f.write(styled_html)
    print(f"✅ Debug HTML saved: {html_file}")
    
    # Generate PDF
    print(f"Generating PDF: {pdf_file}")
    HTML(string=styled_html, base_url=str(Path(md_file).parent)).write_pdf(pdf_file)
    
    print(f"✅ PDF generated successfully!")
    print(f"📄 Output: {pdf_file}")
    
    # Get file size
    size_kb = Path(pdf_file).stat().st_size / 1024
    print(f"📊 Size: {size_kb:.1f} KB")

if __name__ == "__main__":
    md_file = "/Users/umairtalib/DepressiVoice-LLM/docs/reports/Phase1_Comprehensive_Report.md"
    pdf_file = "/Users/umairtalib/DepressiVoice-LLM/docs/reports/Phase1_Comprehensive_Report.pdf"
    
    generate_pdf(md_file, pdf_file)
