pip install hf_xet
pip install -U sentence-transformers
pip install openai
pip install faiss-cpu

from sentence_transformers import SentenceTransformer
import pandas as pd
from openai import OpenAI
import faiss
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def chunk_text(text, chunk_size=500):
    sentences = text.split('\n')
    chunks, current = [], []
    for sentence in sentences:
        if len(' '.join(current)) + len(sentence) < chunk_size:
            current.append(sentence)
        else:
            chunks.append(' '.join(current))
            current = [sentence]
    if current:
        chunks.append(' '.join(current))
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def rerank_local(query, passages, top_n=3):
    pairs = [(query, passage) for passage in passages]
    inputs = rerank_tokenizer.batch_encode_plus(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze(-1)

    sorted_indices = torch.argsort(scores, descending=True)
    reranked_passages = [passages[i] for i in sorted_indices[:top_n]]
    return reranked_passages

def retrieve(query, k=3, rerank_top_n=3):
    print(f"\n[Retrieval] Searching for query: \"{query}\"")
    query_vec = embed_texts([query])
    distances, indices = index.search(np.array(query_vec), k)
    
    retrieved_chunks = chunk_df.iloc[indices[0]].copy()
    print(f"[Retrieval] Top {k} results (pre-rerank distances): {distances[0]}")

    # Rerank
    passages = retrieved_chunks['chunk'].tolist()
    reranked_passages = rerank_local(query, passages, top_n=rerank_top_n)

    print(f"[Retrieval] Top {rerank_top_n} results after reranking.")
    return chunk_df[chunk_df['chunk'].isin(reranked_passages)]

def generate_answer(chunks_df, user_query):
    print("\n[LLM] Preparing prompt for DeepSeek-R1...")
    context = "\n\n".join(chunks_df['chunk'].tolist())
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. Context: {context}
                 Question: {user_query}
                 Answer:"""

    print("[LLM] Sending prompt to DeepSeek-R1...")
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print("[LLM] Response received.")
    return answer


df = pd.read_csv('/content/NZ_repository_scrapped_1.csv')
df.dropna(subset=['text'], inplace=True)

print("Chunking text into smaller segments...")
chunk_rows = []

for _, row in df.iterrows():
    text = row['text']
    if isinstance(text, str):
        chunks = chunk_text(text, chunk_size=1200)
        for i, chunk in enumerate(chunks):
            chunk_rows.append({
                'id': row['id'],
                'title': row['title'],
                'url': row['url'],
                'chunk': chunk
            })

chunk_df = pd.DataFrame(chunk_rows)
print(f"Total chunks created: {len(chunk_df)}")

print("Loading embedding model (this may take a few seconds)...")
model_encode_query = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

print("Generating embeddings for all chunks (this may take a couple of hours)...")
#embeddings = model.encode(chunk_df['chunk'].tolist(), show_progress_bar=True)
chunk_texts = chunk_df['chunk'].iloc[:1000].tolist()
embeddings = embed_texts(chunk_texts, batch_size=16)

print("Creating FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print(f"FAISS index created with {index.ntotal} vectors.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

print("\n🚀 Chatbot is ready! Type your questions below.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chatbot. Goodbye!")
        break

    relevant_chunks = retrieve(user_input, k=3, rerank_top_n=1)
    answer = generate_answer(relevant_chunks, user_input)
    print(f"\nBot: {answer}\n")