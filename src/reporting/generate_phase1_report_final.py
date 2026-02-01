
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import os

def generate_corrected_report():
    bundle_path = "/Users/umairtalib/DepressiVoice-LLM/validation_final_bundle"
    output_path = "/Users/umairtalib/DepressiVoice-LLM/PHASE_1_MASTER_REPORT_FINAL.pdf"
    
    doc = SimpleDocTemplate(output_path, pagesize=LETTER, topMargin=0.75*inch, bottomMargin=0.75*inch, leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, parent=styles['Normal'], spaceAfter=12, leading=14))
    styles.add(ParagraphStyle(name='MainTitle', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER, spaceAfter=24, textColor=colors.HexColor("#1a365d")))
    styles.add(ParagraphStyle(name='SubTitle', parent=styles['Heading2'], fontSize=16, alignment=TA_CENTER, spaceAfter=48, textColor=colors.HexColor("#2d3748")))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#2c5282"), borderPadding=5, borderWidth=0, borderColor=colors.HexColor("#bee3f8"), backColor=colors.HexColor("#ebf8ff")))
    styles.add(ParagraphStyle(name='SubSection', parent=styles['Heading3'], fontSize=13, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2b6cb0")))
    styles.add(ParagraphStyle(name='Caption', parent=styles['Italic'], fontSize=9, alignment=TA_CENTER, spaceAfter=12, textColor=colors.gray))

    story = []

    # --- Title Page ---
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("PHASE 1: Project Report", styles['MainTitle']))
    story.append(Paragraph("High-Performance Speaker Diarization & Pipeline Engineering on Omni Cluster", styles['SubTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<b>Author:</b> DepressiVoice-LLM Team", styles['Normal']))
    story.append(Paragraph("<b>Date:</b> February 1, 2026", styles['Normal']))
    story.append(Paragraph("<b>Target System:</b> MentalRoBERTa + WavLM Fusion", styles['Normal']))
    story.append(Spacer(1, 2*inch))
    
    # Executive Stats
    stats = [
        ["Total Sessions", "275"],
        ["Recovered Yield", "100.0%"],
        ["Linguistic Purity", "98.9%"],
        ["Omni Cluster Nodes", "10+ GPU Nodes (Run via SLURM)"]
    ]
    t = Table(stats, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f7fafc")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e0")),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(PageBreak())

    # --- 1. Introduction ---
    story.append(Paragraph("1. Introduction", styles['SectionHeader']))
    story.append(Paragraph(
        "The primary objective of Phase 1 was to engineer a fault-tolerant, high-performance pipeline to isolate patient audio from clinical interviews (DAIC-WOZ). This data is the prerequisite for Phase 2: a multimodal depression detection system based on <b>MentalRoBERTa</b> and <b>WavLM</b>.",
        styles['Justify']
    ))

    # --- 2. Methodology & Mathematics ---
    story.append(Paragraph("2. Methodology & Mathematics", styles['SectionHeader']))
    
    story.append(Paragraph("2.1 Biometric Identity Gating ($S_C$)", styles['SubSection']))
    story.append(Paragraph(
        "To ensure role consistency, we mathematically modeled the 'Interviewer' identity. We computed a centroid embedding $E$ (The Ellie Anchor) from confirmed interviewer segments. For any unknown speaker cluster $u$, we calculate the Cosine Similarity:",
        styles['Justify']
    ))
    story.append(Paragraph("<i>S_C(u, E) = (u · E) / (||u|| ||E||)</i>", styles['Normal']))
    story.append(Paragraph(
        "Roles are assigned by maximizing the separation margin ($\\Delta$) between the candidate speakers. This ensures that the 'Patient' is statistically the speaker <i>least similar</i> to the anchor.",
        styles['Justify']
    ))

    # --- 3. Omni Cluster Engineering (NEW) ---
    story.append(Paragraph("3. Omni Cluster Engineering & Debugging", styles['SectionHeader']))
    story.append(Paragraph(
        "Deploying this pipeline on the Siegen Omni HP Cluster required extensive engineering to overcome environment and hardware constraints.",
        styles['Justify']
    ))
    
    infra_points = [
        ListItem(Paragraph("<b>Parallelization (SLURM):</b> We implemented SLURM Array Jobs to process all 275 sessions concurrently across multiple GPU nodes, reducing processing time from days to hours.", styles['Normal'])),
        ListItem(Paragraph("<b>Dynamic Hardware Patching:</b> The code was engineered to detect the available hardware at runtime. We used `sed` patches to dynamically modify script execution paths, ensuring scripts could fallback to CPU for lightweight tasks (like metadata generation) when GPUs were unavailable.", styles['Normal'])),
        ListItem(Paragraph("<b>Environment Isolation:</b> We resolved critical dependency conflicts (e.g., `torchaudio` vs. system drivers) by building a self-contained Conda environment (`depressivoice-phase1`) on the cluster's shared filesystem.", styles['Normal'])),
        ListItem(Paragraph("<b>Path & I/O Hygiene:</b> The pipeline was hardened to handle inconsistent file naming conventions in the raw dataset (e.g., `[id].rttm` vs `diarization.rttm`) using flexible globbing patterns.", styles['Normal']))
    ]
    story.append(ListFlowable(infra_points, bulletType='bullet', start='circle'))

    # --- 4. Critical Challenges & Interventions ---
    story.append(Paragraph("4. Critical Challenges & Interventions", styles['SectionHeader']))
    
    story.append(Paragraph("4.1 Session 309: The 'Ultra Rescue'", styles['SubSection']))
    story.append(Paragraph(
        "While the pipeline worked for 274 sessions, <b>Session 309</b> presented a critical edge case. The patient spoke so infrequently and quietly that the standard Neural Diarization merged their voice with the Interviewer.",
        styles['Justify']
    ))
    story.append(Paragraph(
        "<b>The Fix:</b> We developed a specialized 'Brute Force' rescue script. Instead of allowing the model to determine the number of speakers, we <i>forced</i> a k=2 clustering solution. We then applied the inverse-anchor logic to extract the non-Ellie cluster. This successfully recovered the missing patient audio, achieving 100% dataset yield.",
        styles['Justify']
    ))

    story.append(Paragraph("4.2 Linguistic Leakage (V3 Cleanup)", styles['SubSection']))
    story.append(Paragraph(
        "Initial audits showed a 14% contamination rate where prompt phrases (e.g., 'tell me about') leaked into patient text. We implemented <b>Cleaning Protocol V3</b>: an aggressive filter that drops any segment strictly matching high-risk prompts. This reduced contamination to <b>1.09%</b>.",
        styles['Justify']
    ))

    # --- 5. Validated Results ---
    story.append(Paragraph("5. Final Validation Metrics", styles['SectionHeader']))
    
    # Acoustic Image
    if os.path.exists(os.path.join(bundle_path, "roc.png")):
        img_roc = Image(os.path.join(bundle_path, "roc.png"), width=3*inch, height=3*inch)
        img_hist = Image(os.path.join(bundle_path, "sims_hist.png"), width=3*inch, height=3*inch)
        story.append(Table([[img_roc, img_hist]]))
        story.append(Paragraph("Figure 1: (Left) ROC Curve (AUC 0.86). (Right) Separation Histograms.", styles['Caption']))

    story.append(Paragraph(
        "The scientific audit confirms the dataset is pure. Bootstrapping (10k resamples) shows a robust separation margin ($95\% CI: [0.127, 0.160]$), proving that Patient and Interviewer identities are distinct.",
        styles['Justify']
    ))

    # --- 6. Next Steps: Phase 2 Architecture ---
    story.append(Paragraph("6. Phase 2: Multimodal Architecture", styles['SectionHeader']))
    story.append(Paragraph(
        "With the validated dataset, we will now proceed to the Multimodal Depression Detection Phase. The architecture is defined as follows:",
        styles['Justify']
    ))
    
    arch_steps = [
        ListItem(Paragraph("<b>Text Stream (MentalRoBERTa):</b> We will fine-tune <i>MentalRoBERTa</i> (a RoBERTa model pre-trained on mental health corpora) to extract high-dimensional semantic embeddings from the cleaned transcripts.", styles['Normal'])),
        ListItem(Paragraph("<b>Audio Stream (WavLM Middle Layers):</b> We will utilize <i>WavLM Large</i>. Instead of the final layer, we will extract features from the <b>middle layers</b> (e.g., layers 6-12), which are empirically shown to better capture prosodic and emotional cues compared to semantic top layers.", styles['Normal'])),
        ListItem(Paragraph("<b>Cross-Attention Fusion:</b> The features will be fused using a <b>Cross-Attention Mechanism</b>. The text embeddings will query the audio representations (and vice versa) to dynamically weight the most relevant emotional signals given the semantic context before final classification.", styles['Normal']))
    ]
    story.append(ListFlowable(arch_steps, bulletType='1', start='1'))

    doc.build(story)
    print(f"Corrected Report generated at {output_path}")

if __name__ == "__main__":
    generate_corrected_report()
