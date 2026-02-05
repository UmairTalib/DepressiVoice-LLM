#!/usr/bin/env python3
"""
Generate PDF from comprehensive markdown report using weasyprint
"""

import markdown
from pathlib import Path
from weasyprint import HTML, CSS

def generate_pdf(md_file, pdf_file):
    """Convert markdown to PDF"""
    
    print(f"Reading markdown: {md_file}")
    with open(md_file, 'r') as f:
        md_content = f.read()
    
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
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
            }}
        }}
        
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            page-break-before: always;
            margin-top: 0;
        }}
        
        h1:first-of-type {{
            page-break-before: avoid;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 30px;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }}
        
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 8px;
            text-align: left;
            font-size: 10pt;
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
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 5px 0;
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
            margin: 30px 0;
        }}
        
        .page-break {{
            page-break-after: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Generate PDF
    print(f"Generating PDF: {pdf_file}")
    HTML(string=styled_html).write_pdf(pdf_file)
    
    print(f"✅ PDF generated successfully!")
    print(f"📄 Output: {pdf_file}")
    
    # Get file size
    from pathlib import Path
    size_kb = Path(pdf_file).stat().st_size / 1024
    print(f"📊 Size: {size_kb:.1f} KB")

if __name__ == "__main__":
    md_file = "/Users/umairtalib/DepressiVoice-LLM/docs/reports/Phase1_Comprehensive_Report.md"
    pdf_file = "/Users/umairtalib/DepressiVoice-LLM/docs/reports/Phase1_Comprehensive_Report.pdf"
    
    generate_pdf(md_file, pdf_file)
