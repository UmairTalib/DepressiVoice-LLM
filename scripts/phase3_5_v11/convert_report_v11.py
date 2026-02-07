#!/usr/bin/env python3
"""
Convert Phase 3.5 Handover (v11) Markdown Report to HTML using pandoc.
"""

import subprocess
import sys
from pathlib import Path
import os

def convert_md_to_html():
    # Paths
    root = Path(__file__).parent.parent
    md_file = root / "docs/reports/v11_validation/REPORT_v11.md"
    html_file = root / "docs/reports/v11_validation/REPORT_v11.html"
    
    if not md_file.exists():
        print(f"Error: {md_file} not found")
        return False
    
    print(f"Converting {md_file} to HTML...")

    # Method: Pandoc to HTML (Self-contained)
    try:
        # Use simpler pandoc command for HTML
        # Add basic CSS for reasonable look
        css_url = "https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css"
        
        cmd = [
            "pandoc", str(md_file),
            "-o", str(html_file),
            "--standalone",
            "--toc", # Table of Contents
            f"--css={css_url}",
            "--metadata", "title=Phase 3.5 Handover Validation Report (v11)"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and html_file.exists():
            print(f"✓ HTML generated: {html_file}")
            # Patch the HTML to ensure image paths work relative to the HTML file
            # Pandoc usually handles relative paths fine if run from correct dir or absolute paths used
            return True
        else:
            print(f"✗ HTML generation failed: {result.stderr}")
            if "pandoc: command not found" in result.stderr or result.returncode != 0:
                 # Fallback if pandoc missing? 
                 # We can try simple python markdown
                 pass
            return False

    except FileNotFoundError:
        print("pandoc not found. Trying simple python-markdown...")
        try:
            import markdown
            with open(md_file, 'r') as f:
                text = f.read()
            html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
            
            # Wrap in basic HTML structure
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <title>Validation Report v11</title>
            <link rel="stylesheet" href="{css_url}">
            <style>
                body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }}
                @media (max-width: 767px) {{ body {{ padding: 15px; }} }}
            </style>
            </head>
            <body class="markdown-body">
            {html}
            </body>
            </html>
            """
            
            with open(html_file, 'w') as f:
                f.write(full_html)
            print(f"✓ HTML generated (Python-Markdown): {html_file}")
            return True
            
        except ImportError:
            print("python-markdown not found. Please install pandoc or python-markdown.")
            return False

if __name__ == "__main__":
    convert_md_to_html()
