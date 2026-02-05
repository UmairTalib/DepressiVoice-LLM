#!/usr/bin/env python3
"""
Generate Validation Charts for Phase 1 Technical Report
Creates professional visualizations of validation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

def load_validation_data(validation_file):
    """Load validation results"""
    with open(validation_file, 'r') as f:
        return json.load(f)

def create_confidence_pie_chart(data, output_dir):
    """Create pie chart of confidence distribution"""
    
    conf_dist = data['summary']['confidence_distribution']
    
    # Prepare data
    labels = []
    sizes = []
    colors = []
    
    color_map = {
        'HIGH': '#2ecc71',      # Green
        'MEDIUM': '#3498db',    # Blue
        'LOW': '#f39c12',       # Orange
        'VERY_LOW': '#e74c3c',  # Red
        'UNKNOWN': '#95a5a6'    # Gray
    }
    
    for level in ['HIGH', 'MEDIUM', 'LOW', 'VERY_LOW', 'UNKNOWN']:
        count = conf_dist.get(level, 0)
        if count > 0:
            labels.append(f'{level}\n({count} sessions)')
            sizes.append(count)
            colors.append(color_map[level])
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Make percentage text white for visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
    
    ax.set_title('Speaker Diarization Confidence Distribution\n(N=275 sessions)', 
                 fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    output_file = output_dir / 'confidence_distribution.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_file.name}")
    return str(output_file)

def create_margin_histogram(csv_file, output_dir):
    """Create histogram of separation margins"""
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Filter out UNKNOWN confidence (no margin data)
    df_valid = df[df['confidence'] != 'UNKNOWN']
    margins = df_valid['margin'].values
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n, bins, patches = ax.hist(margins, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    
    # Add vertical lines for thresholds
    ax.axvline(0.2, color='#2ecc71', linestyle='--', linewidth=2, label='HIGH threshold (≥0.2)')
    ax.axvline(0.1, color='#3498db', linestyle='--', linewidth=2, label='MEDIUM threshold (≥0.1)')
    ax.axvline(0.05, color='#f39c12', linestyle='--', linewidth=2, label='LOW threshold (≥0.05)')
    
    ax.set_xlabel('Separation Margin (Cosine Similarity Difference)', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Sessions', fontsize=12, weight='bold')
    ax.set_title('Distribution of Speaker Separation Margins\n(N=274 sessions with roles)', 
                 fontsize=14, weight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {margins.mean():.3f}\nMedian: {np.median(margins):.3f}\nStd: {margins.std():.3f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / 'margin_histogram.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_file.name}")
    return str(output_file)

def create_contamination_histogram(csv_file, output_dir):
    """Create histogram of contamination rates"""
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    contamination = df['contamination_rate'].values
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n, bins, patches = ax.hist(contamination, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Highlight low contamination region
    for i, patch in enumerate(patches):
        if bins[i] < 5.0:
            patch.set_facecolor('#2ecc71')
            patch.set_alpha(0.7)
    
    ax.axvline(5.0, color='red', linestyle='--', linewidth=2, label='Acceptable threshold (<5%)')
    
    ax.set_xlabel('Contamination Rate (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Sessions', fontsize=12, weight='bold')
    ax.set_title('Distribution of Transcript Contamination Rates\n(Interviewer prompts in patient transcripts)', 
                 fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {contamination.mean():.2f}%\nMedian: {np.median(contamination):.2f}%\nMax: {contamination.max():.2f}%\n<5%: {(contamination < 5.0).sum()} sessions'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / 'contamination_histogram.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_file.name}")
    return str(output_file)

def create_duration_boxplot(csv_file, output_dir):
    """Create box plot of audio durations"""
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    durations_minutes = df['audio_duration'].values / 60  # Convert to minutes
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bp = ax.boxplot([durations_minutes], vert=True, patch_artist=True,
                     widths=0.5,
                     boxprops=dict(facecolor='#3498db', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax.set_ylabel('Patient Audio Duration (minutes)', fontsize=12, weight='bold')
    ax.set_title('Distribution of Patient Audio Durations\n(N=275 sessions)', 
                 fontsize=14, weight='bold')
    ax.set_xticklabels(['All Sessions'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = (f'Mean: {durations_minutes.mean():.1f} min\n'
                  f'Median: {np.median(durations_minutes):.1f} min\n'
                  f'Q1: {np.percentile(durations_minutes, 25):.1f} min\n'
                  f'Q3: {np.percentile(durations_minutes, 75):.1f} min\n'
                  f'Min: {durations_minutes.min():.1f} min\n'
                  f'Max: {durations_minutes.max():.1f} min\n'
                  f'Total: {durations_minutes.sum()/60:.1f} hours')
    
    ax.text(1.3, 0.5, stats_text, transform=ax.transData,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / 'duration_boxplot.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_file.name}")
    return str(output_file)

def create_quality_flags_chart(csv_file, output_dir):
    """Create bar chart of quality flags"""
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Count flags
    total_flags = df['quality_flag'].sum()
    total_rescue = df['rescue'].sum()
    
    # Create confidence breakdown for flagged sessions
    flagged_df = df[df['quality_flag'] == True]
    conf_counts = flagged_df['confidence'].value_counts()
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Overall flags
    categories = ['Total Sessions', 'Quality Flagged', 'Rescue Sessions']
    counts = [len(df), total_flags, total_rescue]
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    bars1 = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12, weight='bold')
    ax1.set_title('Quality Flag Overview', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Chart 2: Confidence levels of flagged sessions
    levels = conf_counts.index.tolist()
    counts2 = conf_counts.values.tolist()
    color_map = {'HIGH': '#2ecc71', 'MEDIUM': '#3498db', 'LOW': '#f39c12', 'VERY_LOW': '#e74c3c'}
    colors2 = [color_map.get(level, '#95a5a6') for level in levels]
    
    bars2 = ax2.bar(levels, counts2, color=colors2, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Sessions', fontsize=12, weight='bold')
    ax2.set_title('Confidence Levels of Flagged Sessions', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'quality_flags_breakdown.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created: {output_file.name}")
    return str(output_file)

def main():
    # Paths
    downloads_dir = Path('/Users/umairtalib/Downloads')
    validation_json = downloads_dir / 'phase1_validation_results.json'
    validation_csv = downloads_dir / 'phase1_validation_summary.csv'
    
    output_dir = Path('/Users/umairtalib/DepressiVoice-LLM/data/04_results/validation/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GENERATING VALIDATION CHARTS")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading validation data...")
    data = load_validation_data(validation_json)
    
    # Generate charts
    print("\nGenerating charts...\n")
    
    charts = []
    charts.append(create_confidence_pie_chart(data, output_dir))
    charts.append(create_margin_histogram(validation_csv, output_dir))
    charts.append(create_contamination_histogram(validation_csv, output_dir))
    charts.append(create_duration_boxplot(validation_csv, output_dir))
    charts.append(create_quality_flags_chart(validation_csv, output_dir))
    
    print(f"\n{'='*60}")
    print(f"CHART GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated {len(charts)} charts:")
    for chart in charts:
        print(f"  • {Path(chart).name}")
    print(f"\nOutput directory: {output_dir}")
    print()

if __name__ == "__main__":
    main()
