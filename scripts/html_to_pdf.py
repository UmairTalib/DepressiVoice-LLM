#!/usr/bin/env python3
"""
Generate PDF from HTML using Chrome/Chromium headless mode
"""

import subprocess
import shutil
from pathlib import Path

def find_chrome():
    """Find Chrome/Chromium executable"""
    possible_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    
    for path in possible_paths:
        if path and Path(path).exists():
            return path
    return None

def html_to_pdf(html_file, pdf_file):
    chrome_path = find_chrome()
    
    if not chrome_path:
        print("✗ Chrome/Chromium not found")
        print("\nPlease convert manually:")
        print(f"  1. Open {html_file} in your browser")
        print(f"  2. Press Cmd+P (Print)")
        print(f"  3. Select 'Save as PDF'")
        print(f"  4. Save to {pdf_file}")
        return False
    
    print(f"Found Chrome: {chrome_path}")
    print(f"Converting {html_file} to PDF...")
    
    html_path = Path(html_file).resolve()
    pdf_path = Path(pdf_file).resolve()
    
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--print-to-pdf=" + str(pdf_path),
        "--no-pdf-header-footer",
        f"file://{html_path}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if pdf_path.exists():
            print(f"✓ PDF generated successfully: {pdf_path}")
            print(f"  File size: {pdf_path.stat().st_size // 1024} KB")
            return True
        else:
            print(f"✗ PDF generation failed")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Conversion timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert HTML to PDF")
    parser.add_argument("input_html", nargs='?', default="docs/reports/Phase2B_Technical_Report.html", help="Input HTML file path")
    parser.add_argument("output_pdf", nargs='?', default="docs/reports/Phase2B_Technical_Report.pdf", help="Output PDF file path")
    
    args = parser.parse_args()
    
    success = html_to_pdf(args.input_html, args.output_pdf)
    
    if not success:
        print("\n" + "="*60)
        print("ALTERNATIVE OPTION:")
        print("="*60)
        print(f"The HTML report is ready at: {args.input_html}")
        print("\nTo create PDF manually:")
        print("  1. Open the HTML file in Chrome/Safari")
        print("  2. Use Cmd+P to print")
        print("  3. Select 'Save as PDF'")
        print(f"  4. Save to {args.output_pdf}")
