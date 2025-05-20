from config import client

def generate_answer(chunks, query):
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.
Context: {context}
Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
