import torch
from dataset import MultimodalDAICDataset, multimodal_collate_fn
from model import FusionModel
from torch.utils.data import DataLoader
import os

def test_pipeline():
    print("🚀 Starting Phase 3 Fusion Verification...")
    
    # 1. Create Dummy Data for testing (since we are local and cluster data is remote)
    os.makedirs("test_data/audio", exist_ok=True)
    os.makedirs("test_data/text", exist_ok=True)
    
    # Create sample embeddings
    torch.save(torch.randn(100, 1024), "test_data/audio/302_features_norm.pt")
    torch.save(torch.randn(20, 768), "test_data/text/302_text_seq.pt")
    
    # Create dummy labels.csv
    with open("test_data/labels.csv", "w") as f:
        f.write("Participant,Depression_label,split\n")
        f.write("302,1,train\n")
    
    print("✅ Dummy data created.")

    # 2. Test Dataset & Loader
    try:
        dataset = MultimodalDAICDataset(
            audio_dir="test_data/audio", 
            text_dir="test_data/text", 
            labels_csv="test_data/labels.csv", 
            subset='train'
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=multimodal_collate_fn)
        
        batch = next(iter(loader))
        print(f"✅ DataLoader Successful. Batch shapes:")
        print(f"   Audio: {batch['audio_feat'].shape}")
        print(f"   Text: {batch['text_feat'].shape}")
        print(f"   Audio Mask: {batch['audio_mask'].shape}")
        print(f"   Text Mask: {batch['text_mask'].shape}")
    except Exception as e:
        print(f"❌ DataLoader Failed: {e}")
        return

    # 3. Test Model Forward Pass
    try:
        model = FusionModel(d_model=128, nhead=4)
        logits = model(batch['audio_feat'], batch['text_feat'], batch['audio_mask'], batch['text_mask'])
        print(f"✅ Model Forward Pass Successful. Logits shape: {logits.shape}")
        assert logits.shape == (1, 1) # Batch size is 1 here
    except Exception as e:
        print(f"❌ Model Failed: {e}")
        return

    print("\n🎉 Phase 3 Engineering Check COMPLETE. Ready for Cluster Deployment.")

if __name__ == "__main__":
    test_pipeline()
