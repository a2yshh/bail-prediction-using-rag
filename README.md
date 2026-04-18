
# Bail Prediction RAG System

An end-to-end Retrieval-Augmented Generation (RAG) pipeline for predicting bail outcomes in Indian district courts, built on the [IL-TUR dataset](https://huggingface.co/datasets/Exploration-Lab/IL-TUR) with multilingual support via [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B).

-----

## Results

|Metric                          |Score     |
|--------------------------------|----------|
|Retrieval Majority Vote Accuracy|**83.33%**|
|LLM End-to-End Accuracy         |**96.67%**|
|Framework                       |RAGAS     |

-----

## System Architecture

```
Raw Case Text (Hindi / Indic)
        ↓
  IndicTrans2  (ai4bharat/indictrans2-indic-en-1B)
        ↓
  English Translated Text
        ↓
  Sentence Chunking (4 sentences, 1 overlap)
        ↓
  BAAI/bge-base-en-v1.5  Embeddings
        ↓
  ChromaDB  Vector Store
        ↓
  User Query  (structured case details)
        ↓
  Top-K Retrieval  (K=8 similar chunks)
        ↓
  Groq LLaMA3-3.2.instant →  Prediction + Explanation
        ↓
  Streamlit UI

## Dataset

**IL-TUR Bail Dataset** — [Exploration-Lab/IL-TUR](https://huggingface.co/datasets/Exploration-Lab/IL-TUR)

- **Task**: Binary classification — bail `GRANTED` (1) or `DENIED` (0)
- **Language**: Hindi (Devanagari script), district court documents
- **Splits used**: `train_all`, `dev_all`, `test_all`

Each case contains:

- `facts-and-arguments` — sentences from the lawyers’ arguments
- `judge-opinion` — sentences from the judge’s reasoning
- `label` — final bail decision
- `district` — originating district court

-----

## ⚙️ Setup

### 1. Clone and install

```bash
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2 && pip install -r requirements.txt && cd ..

pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
pip install chromadb sentence-transformers groq streamlit \
            python-dotenv ragas datasets langchain-groq
```

### 2. Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

-----

## Running the Pipeline

### Step 1 — Translate the dataset

```bash
python scripts/01_translate.py
```

Translates Hindi case text to English using IndicTrans2 (GPU recommended). Outputs are saved incrementally to `data/translated/` every 50 cases so progress is never lost on interruption.

> Expected time: 2–4 hours for full dataset on a single GPU.  
> Set `MAX_SENTENCES = 2` at the top of the script to do a quick sanity check first.

### Step 2 — Build ChromaDB index

```bash
python scripts/02_build_chromadb.py
```

Chunks translated cases into overlapping sentence windows, embeds them with `BAAI/bge-base-en-v1.5`, and stores them in a persistent ChromaDB collection. Resume-safe — already-indexed chunks are skipped automatically.

### Step 3 — Run the web app

```bash
streamlit run app.py
```

Opens the Streamlit UI in your browser at `http://localhost:8501`.

### Step 4 — Evaluate

```bash
python scripts/04_evaluate.py
```

Runs three evaluation methods and saves results to `data/eval_results.json`.

-----

## 📐 Evaluation Methodology

Three complementary metrics are reported:

### 1. Retrieval Majority Vote Accuracy

For each test case, the facts are embedded and the top-8 most similar chunks are retrieved from ChromaDB. The majority label among retrieved chunks is taken as the prediction and compared to the ground truth. This measures **retrieval quality** independently of the LLM.

### 2. LLM End-to-End Accuracy

The full RAG pipeline (retrieve → prompt → Groq LLaMA3) is run for each test case. The LLM’s final `GRANTED`/`DENIED` prediction is parsed and compared to the ground truth label. This measures **overall system performance**.

### 3. RAGAS Framework Metrics

The [RAGAS](https://github.com/explodinggradients/ragas) library evaluates the RAG pipeline’s internal quality:

|RAGAS Metric         |What it measures                                             |
|---------------------|-------------------------------------------------------------|
|**Faithfulness**     |Does the LLM’s answer stay grounded in the retrieved context?|
|**Answer Relevancy** |Is the generated answer relevant to the user’s query?        |
|**Context Precision**|Are the retrieved chunks actually useful for the query?      |
|**Context Recall**   |Did retrieval capture enough relevant information?           |

-----

## 🌐 Web Application

The Streamlit app provides a structured form for users to input case details:

- **Free-text case description** (Hindi or English — IndicTrans2 handles translation)
- **District court** selection
- **Nature of offence** selection
- **Prior criminal record** toggle
- **Days in custody**

### Output

- Bail prediction with confidence percentage
- Retrieval signal (majority vote across similar cases)
- 3 salient sentences from retrieved cases that drove the decision
- Plain-language legal reasoning explanation
- Expandable view of all retrieved similar cases with similarity scores

-----

## 🔧 Key Configuration

|Parameter      |Location              |Default                |Description                            |
|---------------|----------------------|-----------------------|---------------------------------------|
|`BATCH_SIZE`   |`01_translate.py`     |`8`                    |Sentences per GPU batch for translation|
|`CHUNK_SIZE`   |`02_build_chromadb.py`|`4`                    |Sentences per ChromaDB chunk           |
|`CHUNK_OVERLAP`|`02_build_chromadb.py`|`1`                    |Overlapping sentences between chunks   |
|`TOP_K`        |`03_rag_pipeline.py`  |`8`                    |Retrieved chunks per query             |
|`GROQ_MODEL`   |`03_rag_pipeline.py`  |`llama3-70b-8192`      |Groq model for generation              |
|`EMBED_MODEL`  |`02_build_chromadb.py`|`BAAI/bge-base-en-v1.5`|Sentence embedding model               |

-----

## 🛠️ Tech Stack

|Component   |Technology                                                                                                                    |
|------------|------------------------------------------------------------------------------------------------------------------------------|
|Translation |[IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) + [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)|
|Embeddings  |[sentence-transformers](https://www.sbert.net/) — `BAAI/bge-base-en-v1.5`                                                     |
|Vector Store|[ChromaDB](https://www.trychroma.com/) (persistent, cosine similarity)                                                        |
|LLM         |[Groq](https://groq.com/) — LLaMA3-70B-8192                                                                                   |
|Evaluation  |[RAGAS](https://github.com/explodinggradients/ragas)                                                                          |
|Web UI      |[Streamlit](https://streamlit.io/)                                                                                            |
|Dataset     |[HuggingFace Datasets](https://huggingface.co/datasets/Exploration-Lab/IL-TUR)                                                |

-----

## Disclaimer

This tool is intended for **research and educational purposes only**. It is not a substitute for professional legal advice. Bail decisions involve complex human judgement, and this system’s predictions should not be used as the basis for any legal decisions.

-----

