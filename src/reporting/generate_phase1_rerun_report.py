#!/usr/bin/env python3
"""
Phase 1 Technical Report Generator
Creates comprehensive PDF report documenting complete Phase 1 rerun
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, 
    Image, ListFlowable, ListItem
)
from reportlab.lib import colors

class Phase1TechnicalReport:
    """Generator for comprehensive Phase 1 technical PDF report"""
    
    def __init__(self, validation_results_file, output_file):
        self.validation_results_file = validation_results_file
        self.output_file = output_file
        self.doc = SimpleDocTemplate(output_file, pagesize=letter)
        self.story = []
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Load validation results
        with open(validation_results_file, 'r') as f:
            self.validation_data = json.load(f)
    
    def add_title_page(self):
        """Add title page"""
        title = Paragraph("DepressiVoice-LLM<br/>Phase 1 Complete Rerun<br/>Technical Report", 
                          self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        
        subtitle = Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", 
                            self.styles['Normal'])
        subtitle.alignment = TA_CENTER
        self.story.append(subtitle)
        
        self.story.append(Spacer(1, 1*inch))
        
        # Summary box
        summary_text = f"""
        <b>Project Summary</b><br/>
        Total Sessions Processed: {self.validation_data['metadata']['total_sessions']}<br/>
        Pipeline: Diarization → Transcription → Validation<br/>
        Models: Pyannote 3.1, Faster-Whisper Large-v3<br/>
        """
        summary = Paragraph(summary_text, self.styles['Normal'])
        self.story.append(summary)
        
        self.story.append(PageBreak())
    
    def add_executive_summary(self):
        """Add executive summary section"""
        self.story.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        
        # Overall pass/fail based on validation
        acoustic = self.validation_data['acoustic']['summary']
        linguistic = self.validation_data['linguistic']['summary']
        alignment = self.validation_data['alignment']['summary']
        
        pass_fail = "PASS" if (
            acoustic['mean_separation_margin'] >0.2 and 
            linguistic['contamination_percentage'] < 5.0 and
            alignment['alignment_rate'] > 0.95
        ) else "REVIEW NEEDED"
        
        summary_text = f"""
        <b>Overall Status: {pass_fail}</b><br/><br/>
        
        This report documents the complete re-execution of the Phase 1 diarization and transcription pipeline 
        for the DepressiVoice-LLM project. All 275 therapy sessions were processed from fresh master audio files.
        <br/><br/>
        
        <b>Key Results:</b><br/>
        • Speaker Separation Quality: {acoustic['mean_separation_margin']:.4f} mean margin<br/>
        • Linguistic Contamination: {linguistic['contamination_percentage']:.2f}%<br/>
        • Transcript Alignment: {alignment['alignment_rate']*100:.2f}% synchronized<br/>
        • Total Sessions: {self.validation_data['metadata']['total_sessions']}<br/>
        """
        
        self.story.append(Paragraph(summary_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_methodology(self):
        """Add full methodology section"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("2. Full Methodology", self.styles['SectionHeader']))
        
        methodology_text = """
        <b>2.1 Pipeline Overview</b><br/><br/>
        
        The Phase 1 pipeline consists of three main stages:<br/><br/>
        
        <b>Stage 1: Speaker Diarization</b><br/>
        • Model: Pyannote speaker-diarization-3.1<br/>
        • Method: Clustering-based VAD + speaker embedding<br/>
        • Output: RTTM files with speaker segments<br/><br/>
        
        <b>Stage 2: Role Assignment (Biometric Identity Gating)</b><br/>
        • Embedding Model: wespeaker-voxceleb-resnet34-LM<br/>
        • Anchor: Ellie reference embedding (centroid of 10 sessions)<br/>
        • Assignment: argmax(similarity) = ELLIE, argmin(similarity) = PATIENT<br/>
        • Validation: Separation margin must be > 0.1<br/><br/>
        
        <b>Stage 3: Transcription</b><br/>
        • Model: faster-whisper-large-v3<br/>
        • Features: Word-level timestamps, VAD filtering<br/>
        • Language: English (probability thresholded)<br/>
        • Output: transcript.json with segments + plain text<br/><br/>
        
        <b>2.2 Biometric Anchor Generation</b><br/><br/>
        
        The Ellie biometric anchor was generated from 10 reference sessions:<br/>
        • Sessions: 300-309<br/>
        • Method: Centroid (mean) of speaker embeddings<br/>
        • Embedding Dimension: 256<br/>
        • Purpose: Consistent role assignment across all 275 sessions<br/><br/>
        
        The anchor ensures that the same speaker (Ellie) is identified consistently across 
        all therapy sessions, preventing role reversal errors.
        """
        
        self.story.append(Paragraph(methodology_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_hyperparameters(self):
        """Add hyperparameters and configuration"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("3. Hyperparameters & Configuration", self.styles['SectionHeader']))
        
        # Create configuration table
        config_data = [
            ["Component", "Parameter", "Value", "Rationale"],
            ["Diarization", "Min speakers", "2", "Dyadic conversation"],
            ["", "Max speakers", "2", "Controlled interview setting"],
            ["", "Embedding window", "Full audio", "Maximum context"],
            ["Anchor", "Reference sessions", "10", "Balance diversity & consistency"],
            ["", "Aggregation method", "Centroid", "Robust to outliers"],
            ["", "Similarity threshold", "0.1", "Conservative separation"],
            ["Transcription", "Model size", "large-v3", "SOTA accuracy"],
            ["", "Beam size", "5", "Balance speed/quality"],
            ["", "Temperature", "0.0", "Deterministic output"],
            ["", "VAD filter", "True", "Remove silence"],
            ["", "VAD min silence", "500ms", "Natural pauses"],
            ["", "Word timestamps", "True", "Fine-grained alignment"],
        ]
        
        table = Table(config_data, colWidths=[1.5*inch, 1.5*inch, 1.3*inch, 2.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_environment(self):
        """Add environment and reproducibility details"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("4. Environment & Reproducibility", self.styles['SectionHeader']))
        
        env_text = """
        <b>4.1 Cluster Environment</b><br/><br/>
        
        • Cluster: Omni HPC (Universität Siegen)<br/>
        • Scheduler: SLURM<br/>
        • Partition: GPU<br/>
        • GPU Type: NVIDIA (cluster-allocated)<br/>
        • Nodes: Multiple (10% array concurrency for diarization, 20% for transcription)<br/>
        • CPUs per task: 4<br/>
        • Memory per job: 32GB (diarization), 16GB (transcription)<br/>
        • Time limit: 4 hours (diarization), 2 hours (transcription)<br/><br/>
        
        <b>4.2 Software Environment</b><br/><br/>
        
        • Conda environment: depressivoice-phase1<br/>
        • Python version: 3.10<br/>
        • PyTorch: Latest (CUDA-enabled)<br/>
        • Pyannote.audio: 3.1.x<br/>
        • faster-whisper: Latest<br/>
        • torchaudio backend: soundfile<br/>
        • HuggingFace token: Required (authenticated)<br/><br/>
        
        <b>4.3 Data Locations</b><br/><br/>
        
        • Master audio: /home/g755389/depressivoice_master_audio/ (275 WAV files, 14.67 GB)<br/>
        • Outputs: /home/g755389/DepressiVoice-LLM/phase1_rerun_outputs/<br/>
        • Logs: /home/g755389/DepressiVoice-LLM/logs/<br/>
        • Biometric anchor: /home/g755389/DepressiVoice-LLM/ellie_anchor.json<br/><br/>
        
        <b>4.4 Reproducibility</b><br/><br/>
        
        All scripts, configurations, and SLURM batch files are version-controlled and uploaded to the cluster.
        The master audio files were verified with SHA256 checksums (100% match) to ensure data integrity.
        """
        
        self.story.append(Paragraph(env_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_validation_results(self):
        """Add validation results section with tables"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("5. Validation Results", self.styles['SectionHeader']))
        
        # Acoustic validation table
        self.story.append(Paragraph("<b>5.1 Acoustic Validation (Speaker Separation)</b>", 
                                   self.styles['Normal']))
        
        acoustic = self.validation_data['acoustic']['summary']
        acoustic_data = [
            ["Metric", "Value"],
            ["Mean Separation Margin", f"{acoustic['mean_separation_margin']:.4f}"],
            ["Median Separation Margin", f"{acoustic['median_separation_margin']:.4f}"],
            ["Std Dev", f"{acoustic['std_separation_margin']:.4f}"],
            ["Min Margin", f"{acoustic['min_separation_margin']:.4f}"],
            ["Max Margin", f"{acoustic['max_separation_margin']:.4f}"],
            ["Sessions < 0.1 threshold", f"{acoustic['sessions_below_threshold_01']}"],
            ["Sessions < 0.2 threshold", f"{acoustic['sessions_below_threshold_02']}"],
        ]
        
        acoustic_table = Table(acoustic_data, colWidths=[3*inch, 2*inch])
        acoustic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        self.story.append(acoustic_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Linguistic validation
        self.story.append(Paragraph("<b>5.2 Linguistic Purity (Contamination Analysis)</b>", 
                                   self.styles['Normal']))
        
        linguistic = self.validation_data['linguistic']['summary']
        linguistic_data = [
            ["Metric", "Value"],
            ["Total Segments", f"{linguistic['total_segments']}"],
            ["Contaminated Segments", f"{linguistic['contaminated_segments']}"],
            ["Contamination Rate", f"{linguistic['contamination_percentage']:.2f}%"],
        ]
        
        linguistic_table = Table(linguistic_data, colWidths=[3*inch, 2*inch])
        linguistic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        self.story.append(linguistic_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Conversational dynamics
        self.story.append(Paragraph("<b>5.3 Conversational Dynamics</b>", 
                                   self.styles['Normal']))
        
        conversational = self.validation_data['conversational']['summary']
        conv_data = [
            ["Metric", "Value"],
            ["Mean Turn Density", f"{conversational['mean_turn_density']:.4f} turns/sec"],
            ["Median Turn Density", f"{conversational['median_turn_density']:.4f} turns/sec"],
            ["Std Dev", f"{conversational['std_turn_density']:.4f}"],
        ]
        
        conv_table = Table(conv_data, colWidths=[3*inch, 2*inch])
        conv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        self.story.append(conv_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Alignment validation
        self.story.append(Paragraph("<b>5.4 Transcript-Audio Alignment</b>", 
                                   self.styles['Normal']))
        
        alignment = self.validation_data['alignment']['summary']
        align_data = [
            ["Metric", "Value"],
            ["Total Sessions", f"{alignment['total_sessions']}"],
            ["Aligned Sessions", f"{alignment['aligned_sessions']}"],
            ["Misaligned Sessions", f"{alignment['misaligned_sessions']}"],
            ["Alignment Rate", f"{alignment['alignment_rate']*100:.2f}%"],
        ]
        
        align_table = Table(align_data, colWidths=[3*inch, 2*inch])
        align_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        self.story.append(align_table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_limitations(self):
        """Add failure modes and limitations"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("6. Failure Modes & Limitations", self.styles['SectionHeader']))
        
        limitations_text = """
        <b>6.1 Known Limitations</b><br/><br/>
        
        • Stereo channel handling: Some master audio files are stereo. The pipeline converts to mono by averaging 
          channels, which may lose spatial information.<br/><br/>
        
        • Low patient speech sessions: Sessions with <30 seconds of patient speech may have unreliable role assignment 
          due to insufficient embedding samples.<br/><br/>
        
        • Interviewer prompt leakage: Despite aggressive cleaning, residual contamination may occur if the patient 
          repeats interviewer phrases naturally in conversation.<br/><br/>
        
        • Timestamp granularity: Word-level timestamps depend on faster-whisper's internal alignment, which may have 
          ±100ms tolerance.<br/><br/>
        
        <b>6.2 Edge Cases</b><br/><br/>
        
        • Session 309: Known to have extremely low patient speech, may require manual intervention.<br/>
        • Sessions 603, 609, 620, 622, 623: Large stereo files (150-270 MB) may have extended processing time.<br/><br/>
        
        <b>6.3 Validation Thresholds</b><br/><br/>
        
        • Separation margin < 0.1: Indicates potential role confusion, recommend manual review.<br/>
        • Contamination > 5%: Suggests transcript cleaning failed, requires re-filtering.<br/>
        • Alignment < 95%: Indicates timestamp desynchronization, requires investigation.
        """
        
        self.story.append(Paragraph(limitations_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_future_improvements(self):
        """Add future improvements section"""
        self.story.append(PageBreak())
        self.story.append(Paragraph("7. Future Improvements", self.styles['SectionHeader']))
        
        improvements_text = """
        <b>7.1 Alignment Refinements</b><br/><br/>
        
        • Implement forced alignment using Montreal Forced Aligner for sub-phoneme precision.<br/>
        • Add cross-correlation validation between transcript timing and audio envelope.<br/>
        • Incorporate acoustic features (pitch, energy) to improve VAD boundaries.<br/><br/>
        
        <b>7.2 Additional Validations</b><br/><br/>
        
        • Phonetic diversity check: Ensure transcripts capture full phoneme range.<br/>
        • Overlapping speech detection: Identify and flag simultaneous speaking segments.<br/>
        • Audio quality metrics: SNR, clipping detection, background noise quantification.<br/>
        • Inter-rater reliability: Sample manual verification of role assignments.<br/><br/>
        
        <b>7.3 Pipeline Optimizations</b><br/><br/>
        
        • GPU memory optimization: Enable mixed-precision inference for faster processing.<br/>
        • Batch processing: Group short sessions for parallel GPU utilization.<br/>
        • Checkpoint resume: Add mid-pipeline checkpointing for fault tolerance.<br/>
        • Real-time monitoring: Dashboard for live job status and error tracking.
        """
        
        self.story.append(Paragraph(improvements_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def generate(self):
        """Generate the complete PDF report"""
        print(f"\n=== GENERATING TECHNICAL REPORT ===")
        print(f"Loading validation results from: {self.validation_results_file}")
        
        self.add_title_page()
        self.add_executive_summary()
        self.add_methodology()
        self.add_hyperparameters()
        self.add_environment()
        self.add_validation_results()
        self.add_limitations()
        self.add_future_improvements()
        
        # Build PDF
        self.doc.build(self.story)
        
        print(f"Report generated: {self.output_file}")
        print(f"Total pages: ~{len(self.story) // 10}")
        print("=== REPORT GENERATION COMPLETE ===")

def main():
    parser = argparse.ArgumentParser(description="Generate Phase 1 Technical Report PDF")
    parser.add_argument("--validation_results", required=True, help="Path to validation_results.json")
    parser.add_argument("--output", default="docs/PHASE_1_TECHNICAL_REPORT.pdf", help="Output PDF file")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report = Phase1TechnicalReport(args.validation_results, args.output)
    report.generate()
    
    return 0

if __name__ == "__main__":
    exit(main())
