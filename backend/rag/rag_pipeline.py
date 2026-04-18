import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "bail_cases"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
GROQ_MODEL      = "llama-3.1-8b-instant"   # best free Groq model for reasoning
TOP_K           = 8                   # how many chunks to retrieve

# ─────────────────────────────────────────
# INIT (lazy-loaded so Streamlit doesn't reload every rerun)
# ─────────────────────────────────────────
_embedder   = None
_collection = None
_groq       = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


# ─────────────────────────────────────────
# STEP 1: Retrieve similar chunks
# ─────────────────────────────────────────
def retrieve_similar_chunks(query_text: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embeds the query and retrieves top_k most similar chunks from ChromaDB.
    Returns list of dicts with chunk text, metadata, and distance.
    """
    embedder   = get_embedder()
    collection = get_collection()

    query_vec = embedder.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":      doc,
            "case_id":   meta["case_id"],
            "district":  meta["district"],
            "label":     meta["label"],
            "label_str": meta["label_str"],
            "section":   meta["section"],
            "distance":  dist,
        })

    return chunks


# ─────────────────────────────────────────
# STEP 2: Majority vote prediction from retrieved chunks
# ─────────────────────────────────────────
def majority_vote(chunks: list[dict]) -> tuple[str, float]:
    """
    Simple retrieval-based prediction:
    majority label among retrieved chunks → prediction.
    Returns (label_str, confidence_pct)
    """
    labels = [c["label"] for c in chunks]
    granted = labels.count(1)
    denied  = labels.count(0)
    total   = len(labels)

    if granted >= denied:
        return "GRANTED", round(granted / total * 100, 1)
    else:
        return "DENIED", round(denied / total * 100, 1)


# ─────────────────────────────────────────
# STEP 3: Build prompt and call Groq LLM
# ─────────────────────────────────────────
def build_prompt(user_case: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Case {i+1} | Outcome: {chunk['label_str']} | District: {chunk['district']} | Section: {chunk['section']}]\n"
            f"{chunk['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are an expert Indian legal AI assistant specializing in bail jurisprudence.

You have been provided with the following similar past bail cases retrieved from a database:

{context_str}

---
Now analyze this new case submitted by the user:

{user_case}

---
Based on the similar cases above, provide:

1. PREDICTION: State clearly whether bail is likely to be GRANTED or DENIED.
2. CONFIDENCE: Estimate a confidence percentage (e.g. 70% confident).
3. SALIENT SENTENCES: Identify exactly 3 sentences or phrases from the retrieved cases above that most strongly influenced your prediction. Quote them directly.
4. EXPLANATION: In 3-5 sentences, explain the legal reasoning behind your prediction in plain language that a non-lawyer can understand. Reference specific factors like nature of offence, criminal history, flight risk, evidence strength.

Format your response EXACTLY as follows:
PREDICTION: <GRANTED or DENIED>
CONFIDENCE: <number>%
SALIENT SENTENCES:
- <sentence 1>
- <sentence 2>
- <sentence 3>
EXPLANATION: <your explanation>
"""
    return prompt


def predict_bail(user_case: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve similar chunks
    2. Majority vote
    3. LLM explanation
    Returns structured result dict.
    """
    # 1. Retrieve
    chunks = retrieve_similar_chunks(user_case)

    # 2. Majority vote (fast signal)
    mv_label, mv_confidence = majority_vote(chunks)

    # 3. LLM call
    prompt = build_prompt(user_case, chunks)
    groq   = get_groq()

    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a legal AI assistant that analyzes Indian bail cases. Always follow the exact output format requested."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,   # low temp = more consistent legal reasoning
        max_tokens=1024,
    )

    llm_output = response.choices[0].message.content

    # 4. Parse LLM output
    parsed = parse_llm_output(llm_output)

    return {
        "retrieval_prediction":  mv_label,
        "retrieval_confidence":  mv_confidence,
        "llm_prediction":        parsed["prediction"],
        "llm_confidence":        parsed["confidence"],
        "salient_sentences":     parsed["salient_sentences"],
        "explanation":           parsed["explanation"],
        "retrieved_chunks":      chunks,
        "raw_llm_output":        llm_output,
    }


# ─────────────────────────────────────────
# STEP 4: Parse structured LLM output
# ─────────────────────────────────────────
def parse_llm_output(text: str) -> dict:
    """Parse the structured LLM response into components."""
    result = {
        "prediction": "UNKNOWN",
        "confidence": "N/A",
        "salient_sentences": [],
        "explanation": ""
    }

    lines = text.strip().split("\n")
    mode  = None

    for line in lines:
        line = line.strip()
        if line.startswith("PREDICTION:"):
            val = line.replace("PREDICTION:", "").strip()
            result["prediction"] = "GRANTED" if "GRANT" in val.upper() else "DENIED"
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip()
        elif line.startswith("SALIENT SENTENCES:"):
            mode = "salient"
        elif line.startswith("EXPLANATION:"):
            mode = "explanation"
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
        elif mode == "salient" and line.startswith("-"):
            result["salient_sentences"].append(line[1:].strip())
        elif mode == "explanation" and line:
            result["explanation"] += " " + line

    return result