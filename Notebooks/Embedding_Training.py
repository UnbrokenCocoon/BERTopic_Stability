
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load the BGE model and tokenizer (replace with actual model name)
model_name = "BAAI/bge-base-en"  # or another BGE variant like bge-large-en

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_bge_embeddings(sentences, batch_size=32):
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded = {k: v.cuda() for k, v in encoded.items()}  # Move inputs to GPU

        with torch.no_grad():
            output = model(**encoded)
            embeddings = output.last_hidden_state[:, 0]  # CLS pooling
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalise

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def get_bge_embeddings(sentences, batch_size=32):
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded = {k: v.cuda() for k, v in encoded.items()}  # Move inputs to GPU

        with torch.no_grad():
            output = model(**encoded)
            embeddings = output.last_hidden_state[:, 0]  # CLS pooling
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalise

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

bge_embeddings = get_bge_embeddings(sentences, batch_size=16)
print("Shape:", bge_embeddings.shape)  # (N, 1024)


import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
df = pd.read_parquet("bge_similarity_pairs.parquet")   # or your df
examples = [
    InputExample(texts=[row.sentence1, row.sentence2], label=float(row.similarity))
    for _, row in df.iterrows()
]
train_loader = DataLoader(examples, shuffle=True, batch_size=32)
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
loss = losses.CosineSimilarityLoss(model)
model.fit([(train_loader, loss)], epochs=6, warmup_steps=1000)
model.save("mpnet_distilled_regression")

model = SentenceTransformer("mpnet_distilled_regression", device="cuda")

mpnet_ft = SentenceTransformer("mpnet_distilled_regression")
result_ft = evaluator(mpnet_ft, output_path=None)
print(f"Spearman: {result_ft['spearman_cosine']:.4f} | Pearson: {result_ft['pearson_cosine']:.4f}")



# Optionally split dev/test if needed
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    examples,
    name="baseline-eval",
    main_similarity="cosine",
)

# Load the **untuned** MPNet model

baseline_model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

# Run evaluation
result_base = evaluator(baseline_model, output_path=None)

print(
    f"🔹 Baseline MPNet - Spearman: {result_base['spearman_cosine']:.4f} | "
    f"Pearson: {result_base['pearson_cosine']:.4f}"
)
