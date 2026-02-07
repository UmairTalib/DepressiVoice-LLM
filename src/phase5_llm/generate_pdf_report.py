import json
import os
import re
from fpdf import FPDF

class ClinicalReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, "DepressiVoice | Clinical Diagnostic Summary", ln=True, align="C")
        self.ln(5)
        self.set_draw_color(44, 62, 80)
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()} | AI-Generated Clinical Summary (Confidential)", align="C")

def clean_unicode(text):
    """Replaces Unicode characters that cause fpdf2 errors with standard equivalents."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2022": "-",  # bullet
        "\u2026": "...", # ellipsis
        "\u00b1": "+/-", # plus-minus
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def sanitize_text(text):
    """Bridge for backward compatibility, now just cleans unicode."""
    return clean_unicode(text)

def render_markdown_pdf(pdf, text):
    """Robust markdown parser for FPDF: supports Bold/Italic (***, **, *), Headers (#), and Bullets."""
    import re
    
    # Pre-clean the text for basic consistency
    text = text.replace('\r\n', '\n')
    
    lines = text.split('\n')
    for line in lines:
        raw_line = line.rstrip()
        if not raw_line:
            pdf.ln(4)
            continue
            
        # 1. Headers (### or ## or #)
        if raw_line.lstrip().startswith('#'):
            h_match = re.match(r'^\s*(#+)\s*(.*)', raw_line)
            if h_match:
                level = len(h_match.group(1))
                content = clean_unicode(h_match.group(2))
                # Headers are always Bold
                pdf.set_font("helvetica", "B", 12 if level == 1 else 11)
                pdf.multi_cell(0, 9, content)
                pdf.ln(1)
                continue

        # 2. Bullet points (handles * or - with optional indentation)
        bullet_match = re.match(r'^\s*[\*\-]\s+(.*)', raw_line)
        if bullet_match:
            pdf.set_font("helvetica", "", 10)
            pdf.write(6, "  - ")
            line_content = bullet_match.group(1)
        else:
            line_content = raw_line

        # 3. Inline Formatting (Bold/Italic)
        # Regex to find ***text***, **text**, or *text*
        parts = re.split(r'(\*{1,3}.*?\*{1,3})', line_content)
        
        pdf.set_font("helvetica", "", 10)
        for part in parts:
            if part.startswith('*') and part.endswith('*'):
                style = ""
                if part.startswith('***'):
                    style = "BI"
                    clean_part = part[3:-3]
                elif part.startswith('**'):
                    style = "B"
                    clean_part = part[2:-2]
                else:
                    style = "I"
                    clean_part = part[1:-1]
                
                pdf.set_font("helvetica", style, 10)
                pdf.write(6, clean_unicode(clean_part))
                pdf.set_font("helvetica", "", 10)
            else:
                pdf.write(6, clean_unicode(part))
        
        pdf.ln(6)

def generate_pdf(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    pdf = ClinicalReport()
    pdf.add_page()
    
    # Metadata Header
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(40, 10, f"Participant ID: {data.get('participant_id', 'Unknown')}", ln=True)
    pdf.cell(40, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(5)

    # Narrative Section
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(22, 160, 133)
    pdf.cell(40, 10, "Clinical Narrative & AI Findings", ln=True)
    pdf.ln(2)
    
    pdf.set_text_color(0)
    render_markdown_pdf(pdf, data.get('clinical_narrative', "No narrative provided."))
    
    pdf.output(output_path)
    return output_path

if __name__ == "__main__":
    import sys
    from datetime import datetime
    if len(sys.argv) < 3:
        print("Usage: python generate_pdf_report.py <json_path> <output_path>")
    else:
        generate_pdf(sys.argv[1], sys.argv[2])
