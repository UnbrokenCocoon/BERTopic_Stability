import os
import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

data_dir = r'path/to/your/data/folder'

with open(os.path.join(data_dir, 'bs_sen.pkl'), 'rb') as f:
    sentences = pickle.load(f)
# Load model
model_id = 'BAAI/bge-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()        # move once and set to eval mode


def get_bge_embeddings(sentences, batch_size=32):
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}  # inputs → same device

        with torch.no_grad():
            output = model(**encoded)
            emb = output.last_hidden_state[:, 0]          # CLS
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu())                  # store on CPU

    return torch.cat(all_embeddings, dim=0)


bge_embeddings = get_bge_embeddings(sentences, batch_size=16)
print("Shape:", bge_embeddings.shape)  # (N, 1024)


# Create Database of similarities
def sample_similarity_pairs(embeddings, sentences, num_pairs=1_000_000):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    n = len(sentences)
    sentence1_list = []
    sentence2_list = []
    similarity_list = []

    for _ in tqdm(range(num_pairs), desc="Sampling sentence pairs"):
        i, j = random.sample(range(n), 2)
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]
        sentence1_list.append(sentences[i])
        sentence2_list.append(sentences[j])
        similarity_list.append(sim)

    return sentence1_list, sentence2_list, similarity_list
from tqdm import tqdm


s1, s2, sims = sample_similarity_pairs(bge_embeddings, sentences, num_pairs=100_000)

# Optional: preview with tqdm
for a, b, sim in tqdm(zip(s1[:5], s2[:5], sims[:5]), total=5, desc="Inspecting pairs"):
    print(f"[{sim:.4f}] {a} || {b}"
)
# Save Output

df = pd.DataFrame({"sentence1": s1, "sentence2": s2, "similarity": sims})
df.to_parquet("bge_similarity_pairs.parquet")
