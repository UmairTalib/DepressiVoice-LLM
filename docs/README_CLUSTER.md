# Master Guide: From Blank Cluster to Parallel Analysis

This guide assumes your cluster folder is currently **empty**.

## 1. Upload Your Files
On your **LOCAL** computer (Terminal), run these commands:

1.  **Generate the Code Bundle** (I created a script for this):
    ```bash
    sh create_cluster_bundle.sh
    ```
    *(This creates `depressivoice_cluster_deploy.zip`)*

2.  **Upload the Bundle**:
    ```bash
    scp depressivoice_cluster_deploy.zip <your_user_id>@hpc-login.uni-siegen.de:~/
    ```

3.  **Upload Your Data**:
    ```bash
    # Upload the folder containing your Participant Audio files
    scp -r data/01_raw <your_user_id>@hpc-login.uni-siegen.de:~/data/
    ```

---

## 2. Setup the Cluster (One-Time)
On the **CLUSTER** terminal (SSH in first):

1.  **Unzip the Code**:
    ```bash
    unzip depressivoice_cluster_deploy.zip
    ```

2.  **Check Directory Structure**:
    Your folder should now look like this:
    ```
    ~/
    ├── config.yaml
    ├── requirements.txt
    ├── run_parallel.sh
    ├── src/
    │   ├── data_processing.py
    │   └── merge_results.py
    └── data/
        └── 01_raw/        <-- Your audio files MUST be here
    ```

3.  **Load Python & Create Environment**:
    ```bash
    # Load Python
    module load python/3.11.2

    # Create Virtual Environment
    python3 -m venv venv_gpu_test
    source venv_gpu_test/bin/activate

    # Upgrade pip (Vital step)
    pip install --upgrade pip
    ```

4.  **Install Dependencies** (This uses the cluster's fast internet):
    ```bash
    # Install PyTorch (GPU version)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install WhisperX
    pip install git+https://github.com/m-bain/whisperx.git
    
    # Install Project Libs
    pip install -r requirements.txt
    ```

---

## 3. Run the Parallel Jobs
Everything is configured. Now we launch the army.

1.  **Submit the Job**:
    ```bash
    sbatch run_parallel.sh
    ```
    *This starts 10 jobs. Each job processes 1/10th of your data.*

2.  **Monitor Progress**:
    ```bash
    squeue -u <your_user_id>
    ```
    *(You will see 10 entries running or pending)*

    **To check logs:**
    ```bash
    tail -f logs/parallel_*.log
    ```

---

## 4. Final step: Merge Results
When `squeue` shows NO jobs running (everything finished):

1.  **Combine the chunks**:
    ```bash
    # Make sure env is active
    source venv_gpu_test/bin/activate
    
    python src/merge_results.py
    ```

2.  **Download Results**:
    Back on your **LOCAL** computer:
    ```bash
    scp <your_user_id>@hpc-login.uni-siegen.de:~/data/02_transcribed/transcripts_whisperx_FINAL.csv ./
    ```

**You are done!** 🎯
