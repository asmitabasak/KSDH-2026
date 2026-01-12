import os
import pandas as pd
import torch
from model_utils import BDH_Reasoner
from tqdm import tqdm

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
METADATA_FILE = os.path.join(DATA_DIR, "test_metadata.csv")

def run_inference():
    # Initialize Model
    model = BDH_Reasoner()
    model.eval()
    
    # Load Metadata
    df = pd.read_csv(METADATA_FILE)
    results = []

    print(f"🚀 Starting KDSH Track B Pipeline...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        model.reset_state()
        
        # Phase 1: Priming with Backstory
        # (Assuming backstory is pre-embedded or short enough for one pass)
        backstory_tensor = torch.randn(1, 768) # Placeholder for actual embedding
        _ = model(backstory_tensor, train_synapses=True)
        
        # Phase 2: Continuous Narrative Reasoning (100k+ words)
        novel_path = os.path.join(NOVELS_DIR, row['filename'])
        
        with open(novel_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(5000) # Read 5000 chars at a time
                if not chunk: break
                
                # Convert text chunk to embedding (e.g., using a small BERT or FastText)
                # For this template, we use a placeholder:
                chunk_emb = torch.randn(1, 768) 
                
                # BDH updates its 'belief' state with every chunk
                logits = model(chunk_emb, train_synapses=True)
        
        # Final Decision
        prediction = torch.argmax(logits, dim=-1).item()
        results.append({"id": row['id'], "label": prediction})

    # Save Results
    output_df = pd.DataFrame(results)
    output_df.to_csv("results.csv", index=False)
    print("✅ Inference complete. results.csv generated.")

if __name__ == "__main__":
    run_inference()
