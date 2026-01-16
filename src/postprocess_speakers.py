import pandas as pd
import argparse
import os

def assign_speaker_roles(input_file, output_file):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Ensure we have the necessary columns
    required_cols = ['Participant_ID', 'speaker', 'start_time', 'end_time']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file missing one of required columns: {required_cols}")
        return

    # Calculate duration for each segment
    df['duration'] = df['end_time'] - df['start_time']

    # Initialize a new column for the resolved role
    df['speaker_role'] = 'unknown'

    # Process each participant separately
    participants = df['Participant_ID'].unique()
    print(f"Processing {len(participants)} participants...")

    for pid in participants:
        # Get data for this participant
        p_data = df[df['Participant_ID'] == pid]
        
        # Group by speaker and sum duration
        speaker_stats = p_data.groupby('speaker')['duration'].sum()
        
        # Identify Patient vs Interviewer
        # Logic: Patient speaks the most (Max Duration)
        if len(speaker_stats) > 0:
            patient_label = speaker_stats.idxmax()
            total_duration = speaker_stats.sum()
            patient_duration = speaker_stats.max()
            
            # Print stats for verification
            print(f"Participant {pid}: Patient is {patient_label} ({patient_duration:.1f}s / {total_duration:.1f}s)")
            
            # Apply labels
            # Set everyone to Interviewer first (default)
            df.loc[df['Participant_ID'] == pid, 'speaker_role'] = 'Interviewer'
            # Overwrite the Patient
            df.loc[(df['Participant_ID'] == pid) & (df['speaker'] == patient_label), 'speaker_role'] = 'Patient'
        
    # Check for "unknown" speakers from the previous step and ensure they stay unknown or handled
    # (Though we filtered them out, it's good safety)
    
    # Save result
    print(f"Saving labeled data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done! Columns added: 'speaker_role' (Patient/Interviewer)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign 'Patient' vs 'Interviewer' roles based on speaking duration.")
    parser.add_argument("--input", default="data/02_transcribed/transcripts_whisperx_FINAL.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/02_transcribed/transcripts_whisperx_labeled.csv", help="Output CSV file")
    args = parser.parse_args()

    assign_speaker_roles(args.input, args.output)
