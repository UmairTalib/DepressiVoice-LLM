#!/usr/bin/env python3
"""
Convert Phase 2 Markdown Report to PDF using markdown2 + pdfkit or weasyprint
"""

import subprocess
import sys
from pathlib import Path

def convert_md_to_pdf():
    # Paths
    md_file = Path("docs/reports/Phase2B_Text_Embedding_Report.md")
    pdf_file = Path("docs/reports/Phase2B_Technical_Report.pdf")
    html_file = Path("docs/reports/Phase2B_Technical_Report.html")
    
    print(f"Converting {md_file} to PDF...")
    
    # Try Method 1: Pandoc with wkhtmltopdf
    try:
        print("\nMethod 1: Trying pandoc with wkhtmltopdf...")
        result = subprocess.run([
            "pandoc", str(md_file),
            "-o", str(pdf_file),
            "--pdf-engine=wkhtmltopdf",
            "--toc",
            "-V", "margin-left=1in",
            "-V", "margin-right=1in",
            "-V", "margin-top=1in",
            "-V", "margin-bottom=1in"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and pdf_file.exists():
            print(f"✓ PDF generated successfully: {pdf_file}")
            return True
        else:
            print(f"  Failed: {result.stderr}")
    except FileNotFoundError:
        print("  wkhtmltopdf not found")
    
    # Method 2: Convert to HTML first, then use browser-based PDF
    try:
        print("\nMethod 2: Generating HTML version...")
        result = subprocess.run([
            "pandoc", str(md_file),
            "-o", str(html_file),
            "--standalone",
            "--toc",
            "--css=https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css",
            "-V", "title=Phase 2 Technical Report"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ HTML generated: {html_file}")
            print(f"\nYou can:")
            print(f"  1. Open {html_file} in Chrome/Safari and Print to PDF")
            print(f"  2. Or install wkhtmltopdf: brew install wkhtmltopdf")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✗ PDF generation failed. Please install wkhtmltopdf:")
    print("  brew install wkhtmltopdf")
    return False

if __name__ == "__main__":
    convert_md_to_pdf()
