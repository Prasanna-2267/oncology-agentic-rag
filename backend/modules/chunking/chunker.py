import os
import re
import json
import requests
from PyPDF2 import PdfReader

# -----------------------------
# 🔹 CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"


# -----------------------------
# 🔹 LOAD PDFS
# -----------------------------
def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            documents.append(text)

    return documents


# -----------------------------
# 🔹 FAST SPLIT
# -----------------------------
def split_large_text(text, size=2000):
    return [text[i:i+size] for i in range(0, len(text), size)]


# -----------------------------
# 🔹 LLM AGENTIC CHUNKING
# -----------------------------
def agentic_chunk_document(text):

    prompt = f"""
You are a document segmentation agent.

Task:
Split the document into meaningful chunks.

Rules:
- Each chunk must contain 3–6 sentences
- Do NOT summarize
- Do NOT merge unrelated topics
- Create multiple chunks

Return ONLY JSON list:

[
  {{
    "text": "...",
    "topic": "..."
  }}
]

Document:
{text[:3000]}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=180
        )

        raw = response.json().get("response", "")

        # 🔥 Robust JSON extraction
        match = re.search(r"\[.*\]", raw, re.DOTALL)

        if match:
            json_str = match.group(0)
            json_str = json_str.replace("\n", " ").replace("\r", "")

            parsed = json.loads(json_str)
            return parsed

        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print("⚠️ LLM chunking failed, fallback used:", e)
        return []


# -----------------------------
# 🔹 FALLBACK CHUNKING
# -----------------------------
def fallback_chunk(text):

    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < 1000:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks


# -----------------------------
# 🔹 MAIN PIPELINE
# -----------------------------
def chunk_text(raw_docs):

    all_chunks = []

    for doc in raw_docs:

        windows = split_large_text(doc)

        for i, w in enumerate(windows):

            # 🔥 Only first 2 windows use LLM
            if i < 2:
                llm_chunks = agentic_chunk_document(w)

                if llm_chunks:
                    for c in llm_chunks:
                        text = c.get("text", "").strip()

                        if len(text.split()) > 40:
                            all_chunks.append(text)
                else:
                    all_chunks.extend(fallback_chunk(w))

            else:
                all_chunks.extend(fallback_chunk(w))

    return all_chunks