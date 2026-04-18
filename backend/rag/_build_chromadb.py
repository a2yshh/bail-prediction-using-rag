import json
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
TRANSLATED_DIR  = Path("data/translated")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "bail_cases"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"   # strong legal text embedder
CHUNK_SIZE      = 4    # sentences per chunk
CHUNK_OVERLAP   = 1    # overlapping sentences between chunks

# Which splits to index — add "train" and "test" once they're done
SPLITS_TO_INDEX = ["dev"]   # ← change to ["dev","train","test"] later

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
print("Embedding model loaded.")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  
)

print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")

def chunk_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a list of sentences into overlapping chunks.
    Each chunk is a single joined string.
    """
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks

total_chunks = 0

for split_name in SPLITS_TO_INDEX:
    file_path = TRANSLATED_DIR / f"{split_name}.json"
    if not file_path.exists():
        print(f"⚠ {split_name}.json not found, skipping.")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\nIndexing split: {split_name} ({len(cases)} cases)")

    for idx, case in enumerate(cases):
        case_id  = case["id"]
        district = case["district"]
        label    = case["label"]        # 0 or 1
        label_str = case["label_str"]   # "DENIED" or "GRANTED"

        facts   = case["text"]["facts-and-arguments"]
        opinion = case["text"]["judge-opinion"]

        # Chunk each section separately so we know where the chunk came from
        fact_chunks    = chunk_sentences(facts,   CHUNK_SIZE, CHUNK_OVERLAP)
        opinion_chunks = chunk_sentences(opinion, CHUNK_SIZE, CHUNK_OVERLAP)

        all_chunks = []
        all_ids    = []
        all_meta   = []

        for c_idx, chunk_text in enumerate(fact_chunks):
            chunk_id = f"{split_name}_{case_id}_facts_{c_idx}"
            all_chunks.append(chunk_text)
            all_ids.append(chunk_id)
            all_meta.append({
                "case_id":  case_id,
                "district": district,
                "label":    label,
                "label_str": label_str,
                "section":  "facts-and-arguments",
                "split":    split_name,
            })

        for c_idx, chunk_text in enumerate(opinion_chunks):
            chunk_id = f"{split_name}_{case_id}_opinion_{c_idx}"
            all_chunks.append(chunk_text)
            all_ids.append(chunk_id)
            all_meta.append({
                "case_id":  case_id,
                "district": district,
                "label":    label,
                "label_str": label_str,
                "section":  "judge-opinion",
                "split":    split_name,
            })

        # Skip if all chunks already exist in ChromaDB (for resuming)
        existing = collection.get(ids=all_ids)
        existing_ids = set(existing["ids"])
        new_indices = [i for i, cid in enumerate(all_ids) if cid not in existing_ids]

        if not new_indices:
            continue

        new_chunks = [all_chunks[i] for i in new_indices]
        new_ids    = [all_ids[i]    for i in new_indices]
        new_meta   = [all_meta[i]   for i in new_indices]

        # Embed
        embeddings = embedder.encode(new_chunks, show_progress_bar=False).tolist()

        # Upsert into ChromaDB
        collection.upsert(
            ids=new_ids,
            embeddings=embeddings,
            documents=new_chunks,
            metadatas=new_meta,
        )

        total_chunks += len(new_indices)

        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(cases)}] cases indexed | total chunks so far: {total_chunks}")

print(f"\n✅ Indexing complete. Total chunks in ChromaDB: {collection.count()}")