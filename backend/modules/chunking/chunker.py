import re
import requests
import os
from PyPDF2 import PdfReader

# -----------------------------------
# 🔹 CONFIG
# -----------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"


# -----------------------------------
# 🔹 LOAD PDFS
# -----------------------------------
def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))

            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"

            documents.append({
                "id": file,     # 📄 file-level id
                "text": text
            })

    return documents


# -----------------------------------
# 🔹 FALLBACK CHUNKING
# -----------------------------------
def fallback_chunk(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < max_chunk_size:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


# -----------------------------------
# 🔹 LLM CHUNKING
# -----------------------------------
def llm_chunk(text):

    prompt = f"""
Split the following medical text into meaningful semantic chunks.

Rules:
- Each chunk should be self-contained
- Keep related sentences together
- DO NOT return JSON
- Use this format strictly:

[CHUNK]
text...
[/CHUNK]

Text:
{text[:3000]}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 500,
                    "keep_alive": "10m"
                }
            },
            timeout=60
        )

        raw = response.json().get("response", "")

        # 🔹 Extract chunks
        chunks = re.findall(r'\[CHUNK\](.*?)\[/CHUNK\]', raw, re.DOTALL)

        chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

        if len(chunks) > 0:
            return chunks
        else:
            raise ValueError("No chunks extracted")

    except Exception as e:
        print("⚠️ LLM chunking failed, fallback used:", e)
        return None


# -----------------------------------
# 🔹 MAIN CHUNKING PIPELINE
# -----------------------------------
def chunk_text(documents):

    final_chunks = []
    chunk_counter = 0  # 🔥 IMPORTANT

    for i, doc in enumerate(documents):
        text = doc["text"]
        file_id = doc["id"]

        print(f"📄 Processing doc {i+1}/{len(documents)}: {file_id}")

        # 🔥 Hybrid strategy
        if i < 2:
            chunks = llm_chunk(text)
            if chunks is None:
                chunks = fallback_chunk(text)
        else:
            chunks = fallback_chunk(text)

        # 🔥 Assign UNIQUE chunk IDs
        for c in chunks:
            if len(c.split()) < 10:
                continue  # skip very small chunks

            final_chunks.append({
                "id": f"doc_{chunk_counter}",   # ✅ UNIQUE ID
                "text": c,
                "doc_id": file_id              # 📄 source file
            })

            chunk_counter += 1

    # 🔍 DEBUG CHECK
    print("\n🔍 SAMPLE CHUNKS:")
    for c in final_chunks[:3]:
        print(c)

    print(f"\n✅ Total chunks created: {len(final_chunks)}")

    return final_chunks