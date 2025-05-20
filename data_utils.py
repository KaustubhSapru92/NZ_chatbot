import pandas as pd

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

def load_and_chunk_csv(csv_path, chunk_size=1200):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['text'], inplace=True)
    chunk_rows = []
    for _, row in df.iterrows():
        if isinstance(row['text'], str):
            chunks = chunk_text(row['text'], chunk_size)
            for chunk in chunks:
                chunk_rows.append({
                    'id': row['id'],
                    'title': row['title'],
                    'url': row['url'],
                    'chunk': chunk
                })
    return pd.DataFrame(chunk_rows)