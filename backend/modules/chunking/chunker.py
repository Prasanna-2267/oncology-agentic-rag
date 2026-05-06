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

                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

            documents.append({
                "id": file,
                "text": text
            })

    return documents


# -----------------------------------
# 🔹 TEXT CLEANING
# -----------------------------------
def clean_text(text):

    # remove repeated spaces
    text = re.sub(r'\s+', ' ', text)

    # remove page numbers alone
    text = re.sub(r'\b\d+\b', ' ', text)

    # remove excessive punctuation
    text = re.sub(r'[-=_]{2,}', ' ', text)

    # remove weird unicode artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()


# -----------------------------------
# 🔹 CHUNK QUALITY FILTER
# -----------------------------------
def is_good_chunk(text):

    text_lower = text.lower()

    # 🔥 Too short
    if len(text.split()) < 40:
        return False

    # 🔥 Too many digits
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    if digit_ratio > 0.25:
        return False

    # 🔥 Too many uppercase chars
    upper_ratio = sum(c.isupper() for c in text) / max(len(text), 1)

    if upper_ratio > 0.35:
        return False

    # 🔥 Garbage / noisy sections
    bad_patterns = [
        "references",
        "bibliography",
        "isbn",
        "doi",
        "www.",
        "http",
        "copyright",
        "all rights reserved",
        "table of contents",
        "acknowledgment",
        "appendix",
        "index",
        "figure",
        "fig.",
        "publication",
        "editor",
        "authors",
        "downloaded from",
    ]

    for pattern in bad_patterns:
        if pattern in text_lower:
            return False

    # 🔥 Excessive line breaks
    if text.count("\n") > 15:
        return False

    # 🔥 OCR garbage detection
    weird_chars = sum(
        1 for c in text
        if not (
            c.isalnum()
            or c.isspace()
            or c in ".,:;-()%/"
        )
    )

    weird_ratio = weird_chars / max(len(text), 1)

    if weird_ratio > 0.08:
        return False

    # 🔥 Very repetitive chunks
    words = text_lower.split()

    unique_ratio = len(set(words)) / max(len(words), 1)

    if unique_ratio < 0.35:
        return False

    return True


# -----------------------------------
# 🔹 FALLBACK CHUNKING
# -----------------------------------
def fallback_chunk(text, max_chunk_size=700):

    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for sent in sentences:

        sent = sent.strip()

        if not sent:
            continue

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
# 🔹 LLM SEMANTIC CHUNKING
# -----------------------------------
def llm_chunk(text):

    prompt = f"""
Split the following medical text into meaningful semantic chunks.

Rules:
- Each chunk should be self-contained
- Keep medically related sentences together
- Preserve medical meaning
- Do NOT summarize
- Do NOT explain
- Use this exact format:

[CHUNK]
text...
[/CHUNK]

TEXT:
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
                    "temperature": 0.1,
                    "num_predict": 600,
                    "keep_alive": "10m"
                }
            },
            timeout=90
        )

        raw = response.json().get("response", "")

        chunks = re.findall(
            r'\[CHUNK\](.*?)\[/CHUNK\]',
            raw,
            re.DOTALL
        )

        chunks = [
            clean_text(c.strip())
            for c in chunks
            if len(c.strip()) > 50
        ]

        if len(chunks) > 0:
            return chunks

        raise ValueError("No chunks extracted")

    except Exception as e:

        print("⚠️ LLM chunking failed, fallback used:", e)

        return None


# -----------------------------------
# 🔹 MAIN CHUNKING PIPELINE
# -----------------------------------
def chunk_text(documents):

    final_chunks = []

    chunk_counter = 0

    for i, doc in enumerate(documents):

        text = clean_text(doc["text"])

        file_id = doc["id"]

        print(f"📄 Processing doc {i+1}/{len(documents)}: {file_id}")

        # -----------------------------------
        # 🔹 Hybrid Chunking Strategy
        # -----------------------------------
        if i < 2:

            chunks = llm_chunk(text)

            if chunks is None:
                chunks = fallback_chunk(text)

        else:
            chunks = fallback_chunk(text)

        # -----------------------------------
        # 🔹 Filtering + Metadata
        # -----------------------------------
        for c in chunks:

            c = c.strip()

            if not is_good_chunk(c):
                continue

            final_chunks.append({
                "id": f"doc_{chunk_counter}",
                "text": c,
                "doc_id": file_id
            })

            chunk_counter += 1

    # -----------------------------------
    # 🔹 DEBUG
    # -----------------------------------
    print("\n🔍 SAMPLE CHUNKS:\n")

    for c in final_chunks[:3]:
        print(c)
        print()

    print(f"\n✅ Total cleaned chunks created: {len(final_chunks)}")

    return final_chunks