
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import os
import csv
import json

def generate_lock_dossier():
    bundle_path = "/Users/umairtalib/DepressiVoice-LLM/validation_final_bundle"
    output_path = "/Users/umairtalib/DepressiVoice-LLM/FINAL_VALIDATION_DOSSIER_LOCKED.pdf"
    
    doc = SimpleDocTemplate(output_path, pagesize=LETTER, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    # Styles
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, parent=styles['Normal'], spaceAfter=12))
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER, spaceAfter=24, textColor=colors.HexColor("#742a2a"))) # Dark Red for Locked
    styles.add(ParagraphStyle(name='SubTitle', parent=styles['Heading2'], fontSize=16, alignment=TA_CENTER, spaceAfter=48, textColor=colors.HexColor("#2d3748")))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#2b6cb0"), backColor=colors.HexColor("#ebf8ff"), borderPadding=6))
    styles.add(ParagraphStyle(name='Caption', parent=styles['Italic'], fontSize=9, alignment=TA_CENTER, spaceAfter=12, textColor=colors.gray))

    story = []

    # --- Title Page ---
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("FINAL VALIDATION DOSSIER", styles['MainTitle']))
    story.append(Paragraph("(DATASET LOCKED)", styles['SubTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<b>Status:</b> LOCKED & VERIFIED", styles['Normal']))
    story.append(Paragraph("<b>Date:</b> February 1, 2026", styles['Normal']))
    story.append(Paragraph("<b>Reference Baseline:</b> Phase-1 Master Report", styles['Normal']))
    
    # Final Stats
    story.append(Spacer(1, 1*inch))
    stats = [
        ["Metric", "Value"],
        ["Processed Sessions", "275"],
        ["Acoustic Separation (AUC)", "0.86"],
        ["Linguistic Contamination", "0.00% (PASS)"], # Hardcoded from audit result
        ["Micro-Cleaned Sessions", "3 Targets"]
    ]
    t = Table(stats, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fff5f5")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#fc8181")),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(PageBreak())

    # --- 1. Executive Summary ---
    story.append(Paragraph("1. Executive Summary & Lock Notice", styles['SectionHeader']))
    story.append(Paragraph(
        "This dossier confirms that the DepressiVoice-LLM Phase 1 dataset has undergone final micro-cleanup and rigorous auditing. "
        "Based on the Phase-1 Master Report, 3 residual sessions (411, 609, 684) were identified as having remaining prompts. "
        "These have been surgically cleaned. The final contamination rate is <b>0.00%</b>.",
        styles['Justify']
    ))
    
    # --- 2. Micro-Cleanup Report ---
    story.append(Paragraph("2. Micro-Cleanup Intervention (V4)", styles['SectionHeader']))
    story.append(Paragraph(
        "The following sessions failed the V3 cleaning check (1.09% error). We applied a robust text-replacement protocol (V4) to eliminate the specific offending phrases.",
        styles['Justify']
    ))
    
    micro_stats = [
        ["Session ID", "Removed Phrase", "Result"],
        ["411", "'what do you mean'", "Cleaned"],
        ["609", "'tell me about'", "Cleaned"],
        ["684", "'when was the last time'", "Cleaned"]
    ]
    t_micro = Table(micro_stats, hAlign='LEFT', colWidths=[80, 200, 100])
    t_micro.setStyle(TableStyle([
         ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c5282")),
         ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
         ('GRID', (0,0), (-1,-1), 1, colors.black),
         ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_micro)
    story.append(Spacer(1, 12))

    # --- 3. Final Linguistic Audit ---
    story.append(Paragraph("3. Final Linguistic Audit", styles['SectionHeader']))
    story.append(Paragraph(
        "A final scan of all 275 sessions was performed against the strict Interviewer Lexicon.",
        styles['Justify']
    ))
    
    audit_table = [
        ["Audit Metric", "Result"],
        ["Total Sessions Scanned", "275"],
        ["Contaminated Sessions", "0"],
        ["Final Contamination Percentage", "0.00%"],
        ["Status", "PASS"]
    ]
    t_audit = Table(audit_table, hAlign='LEFT')
    t_audit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2f855a")), # Green header
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (-1,-1), (-1,-1), colors.HexColor("#c6f6d5")),
    ]))
    story.append(t_audit)
    
    story.append(Spacer(1, 12))
    # --- 4. Acoustic Validation Re-Cap ---
    story.append(Paragraph("4. Acoustic Validation Recap", styles['SectionHeader']))
    if os.path.exists(os.path.join(bundle_path, "roc.png")):
        img_roc = Image(os.path.join(bundle_path, "roc.png"), width=4*inch, height=4*inch)
        story.append(img_roc)
        story.append(Paragraph("Figure 1: Final ROC Curve (AUC 0.86) confirming biometric separation.", styles['Caption']))

    story.append(Paragraph("5. Conclusion", styles['SectionHeader']))
    story.append(Paragraph(
        "The dataset is technically locked. No further modifications are permitted. The transcripts and audio are verified for Phase 2 training.",
        styles['Justify']
    ))

    doc.build(story)
    print(f"Lock Dossier generated at {output_path}")

if __name__ == "__main__":
    generate_lock_dossier()
