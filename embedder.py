from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading embedding model once...")
embedder_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)


# -----------------------------
# 2. Mean Pooling Function
# -----------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# -----------------------------
# 3. Embedding Function
# -----------------------------
def embed_texts(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = embedder_tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = embedder_model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)