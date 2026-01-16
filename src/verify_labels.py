import pandas as pd
from collections import Counter

def verify_labels(filename):
    print(f"Verifying labels in {filename}...\n")
    df = pd.read_csv(filename)
    
    # Check 1: Interviewer Vocabulary
    # The interviewer (Ellie) is a bot with limited vocabulary.
    # Her top sentences should be short confirmations.
    interviewer_df = df[df['speaker_role'] == 'Interviewer']
    patient_df = df[df['speaker_role'] == 'Patient']
    
    print("--- CHECK 1: Top 10 Phrases by 'Interviewer' (Should be 'Okay', 'I see', etc) ---")
    if not interviewer_df.empty:
        # Normalize text
        text_counts = Counter(interviewer_df['transcript'].str.lower().str.strip())
        for phrase, count in text_counts.most_common(10):
            print(f"{count:5d} x : '{phrase}'")
    else:
        print("WARNING: No Interviewer rows found!")

    # Check 2: Turn Length
    # Patients usually speak more words per turn
    print("\n--- CHECK 2: Average Word Count per Turn ---")
    avg_words_interviewer = interviewer_df['transcript'].str.split().str.len().mean()
    avg_words_patient = patient_df['transcript'].str.split().str.len().mean()
    
    print(f"Interviewer Avg Words/Turn: {avg_words_interviewer:.2f}")
    print(f"Patient     Avg Words/Turn: {avg_words_patient:.2f}")
    
    if avg_words_patient > avg_words_interviewer:
        print("\n✅ PASSED: Patients talk more than the Interviewer.")
    else:
        print("\n❌ WARNING: Patient word count is unusually low.")

    # Check 3: The "Golden Question"
    # "How are you doing today" is the standard opening line.
    print("\n--- CHECK 3: Who asks 'How are you doing today?' ---")
    golden_phrases = df[df['transcript'].str.lower().str.contains("how are you doing today")]
    
    if not golden_phrases.empty:
        assigned_roles = golden_phrases['speaker_role'].unique()
        print(f"This phrase was spoken by roles: {assigned_roles}")
        if 'Patient' in assigned_roles:
             print("❌ WARNING: A 'Patient' asked how you are doing today? (Possible Label Flip)")
        else:
             print("✅ PASSED: Only 'Interviewer' asked this.")
    else:
        print("Note: Exact phrase 'how are you doing today' not found in text.")

if __name__ == "__main__":
    verify_labels("data/02_transcribed/transcripts_whisperx_labeled.csv")
