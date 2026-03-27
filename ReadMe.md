# 🏥 Medical Named Entity Recognition (NER)
### Clinical Entity Extractor — PubMedBERT + Custom CRF with Viterbi Decoding

> An end-to-end **Biomedical NLP system** that extracts **diseases** and **chemicals/drugs** from clinical text — built by fine-tuning Microsoft's PubMedBERT with a **hand-implemented CRF layer and Viterbi decoder** (built entirely from scratch using PyTorch), trained on the BC5CDR benchmark dataset, and deployed as an interactive Streamlit web application.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Why This Project is Technically Challenging](#-why-this-project-is-technically-challenging)
- [Solution Architecture](#-solution-architecture)
- [Tech Stack](#-tech-stack)
- [Model Architecture — Deep Dive](#-model-architecture--deep-dive)
- [Dataset](#-dataset)
- [NER Label Scheme (BIO Tagging)](#-ner-label-scheme-bio-tagging)
- [Key Algorithms Implemented from Scratch](#-key-algorithms-implemented-from-scratch)
- [Inference Pipeline — Sliding Window](#-inference-pipeline--sliding-window)
- [Project Structure](#-project-structure)
- [How to Run Locally](#-how-to-run-locally)
- [Application Workflow](#-application-workflow)
- [Key Results & Metrics](#-key-results--metrics)
- [Skills Demonstrated](#-skills-demonstrated)
- [Author](#-author)

---

## 🧭 Project Overview

Medical literature — patient reports, research abstracts, discharge summaries — contains critical clinical entities like **disease names** and **drug/chemical mentions** buried inside unstructured text. Manually extracting these at scale is impractical. This project automates that extraction using state-of-the-art **Biomedical NLP**.

The system takes any medical text as input, runs it through a fine-tuned **PubMedBERT + CRF** model, and outputs a clean, categorized list of:
- 🔴 **Diseases** — e.g., *diabetic retinopathy*, *Parkinson's disease*
- 🟢 **Chemicals / Drugs** — e.g., *metformin*, *lithium carbonate*

What makes this project stand out technically is that the **CRF (Conditional Random Field) layer and Viterbi decoding algorithm were implemented completely from scratch in PyTorch** — no external CRF libraries used — demonstrating a deep understanding of sequence labeling theory and neural network design.

---

## ❗ Problem Statement

- Biomedical literature grows at millions of papers per year — manual annotation is infeasible
- Generic NLP models (standard BERT, SpaCy) perform poorly on biomedical text due to domain-specific terminology
- Identifying precise spans of diseases and chemicals in complex clinical sentences requires **sequence-aware** models that understand word dependencies, not just individual word probabilities
- **Goal:** Build a high-accuracy, domain-adapted NER pipeline that correctly identifies and classifies clinical entity spans, handles long medical texts, and is accessible via a user-friendly interface

---

## 💡 Why This Project is Technically Challenging

| Challenge | How It Was Solved |
|-----------|------------------|
| Standard BERT tokenizes words into subwords (e.g., *metformin* → *met*, *##form*, *##in*), breaking entity spans | Custom subword reassembly logic in post-processing aligns tokens back to words |
| Medical texts often exceed BERT's 512-token limit | **Sliding window with stride=100** splits long texts into overlapping chunks for complete coverage |
| Simple softmax classification ignores label dependencies (e.g., `I-Disease` cannot legally follow `B-Chemical`) | **Custom CRF layer** enforces valid label transition constraints |
| CRF decoding at inference needs the globally optimal label sequence, not just greedy local choices | **Viterbi algorithm** implemented from scratch finds the exact optimal path |
| Domain mismatch: standard BERT trained on Wikipedia/BookCorpus has no biomedical vocabulary | Used **Microsoft PubMedBERT** — pre-trained exclusively on PubMed abstracts and full-text articles |
| Stop words can appear at entity boundaries after subword reassembly | Post-processing stop-word cleanup trims entity strings for clean output |

---

## 🔧 Solution Architecture

```
Input Medical Text
        ↓
[ Sliding Window Tokenization ]
  (max_length=512, stride=100)
  → Handles texts of any length
        ↓
[ PubMedBERT Encoder ]
  (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
  → Contextual embeddings of size 768 per token
        ↓
[ Dropout Layer (p=0.1) ]
  → Regularization
        ↓
[ Linear Classifier ]
  (768 → 5 emission scores per token)
  → Raw scores for each BIO label
        ↓
[ Custom CRF Layer ]
  → Enforces valid label transitions via learned transition matrix
  → At training: computes NLL loss using Forward Algorithm
  → At inference: runs Viterbi decoding for optimal label path
        ↓
[ Post-Processing ]
  → Subword reassembly (## tokens merged)
  → Entity span extraction (B- / I- tag grouping)
  → Stop-word boundary cleanup
  → Deduplication across sliding window chunks
        ↓
Output: { Diseases: [...], Chemicals/Drugs: [...] }
```

---

## 🛠 Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| **Programming Language** | Python 3.11+ |
| **Deep Learning Framework** | PyTorch |
| **Transformer Model** | HuggingFace Transformers (`AutoTokenizer`, `AutoModel`, `AutoConfig`) |
| **Pre-trained Model** | Microsoft BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext |
| **Model Hosting** | HuggingFace Hub (`hf_hub_download`) |
| **Sequence Labeling** | Custom CRF + Viterbi Decoder (built from scratch in PyTorch) |
| **Data Processing** | NumPy |
| **Web Application** | Streamlit |
| **Development Environment** | Jupyter Notebook, Google Colab |
| **Version Control** | Git, GitHub |

---

## 🧠 Model Architecture — Deep Dive

The model is a **two-component neural architecture** — a Transformer encoder feeding into a Conditional Random Field:

### Component 1: `PubMedBERT_CRF` (Main Model)

```python
class PubMedBERT_CRF(nn.Module):
    def __init__(self, model_name, num_labels):
        self.bert       = AutoModel.from_pretrained(model_name)   # Biomedical BERT encoder
        self.dropout    = nn.Dropout(0.1)                          # Regularization
        self.classifier = nn.Linear(hidden_size=768, num_labels=5) # Emission score generator
        self.crf        = PureCRF(num_labels)                      # Custom CRF head
```

- At **training**: BERT encodes tokens → Linear generates emission scores → CRF computes NLL loss using the **Forward Algorithm** (log-sum-exp over all possible label paths)
- At **inference**: BERT encodes tokens → Linear generates emission scores → CRF runs **Viterbi decoding** to find the single best global label sequence

### Component 2: `PureCRF` (Built from Scratch)

Three learnable parameter matrices:
- `transitions[i][j]` — score of transitioning from label `j` to label `i`
- `start_transitions[i]` — score of a sequence starting with label `i`
- `end_transitions[i]` — score of a sequence ending with label `i`

All initialized with `uniform_(-0.1, 0.1)` and learned during fine-tuning on BC5CDR.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | BC5CDR (BioCreative V Chemical-Disease Relation) |
| **Source** | [BioCreative Challenge / HuggingFace Datasets](https://huggingface.co/datasets/tner/bc5cdr) |
| **Domain** | PubMed biomedical abstracts |
| **Entity Types** | Disease, Chemical |
| **Annotation Format** | BIO tagging scheme (token-level) |
| **Use Case** | Standard NER benchmark for biomedical NLP |

---

## 🏷 NER Label Scheme (BIO Tagging)

The model predicts one of **5 labels** per token, following the BIO (Beginning-Inside-Outside) tagging convention:

| Label | ID | Meaning | Example Token |
|-------|-----|---------|--------------|
| `O` | 0 | Outside any entity | *"The"*, *"patient"*, *"was"* |
| `B-Chemical` | 1 | Beginning of a chemical/drug entity | *"metformin"* |
| `B-Disease` | 2 | Beginning of a disease entity | *"diabetic"* |
| `I-Disease` | 3 | Inside (continuation of) a disease entity | *"retinopathy"* |
| `I-Chemical` | 4 | Inside (continuation of) a chemical entity | *"##ate"* in *"carbonate"* |

**Example annotation:**

```
Token:   The   patient   was   diagnosed   with   diabetic      retinopathy   and   metformin
Label:    O       O        O       O          O    B-Disease      I-Disease     O    B-Chemical
```

---

## ⚙ Key Algorithms Implemented from Scratch

### 1. Forward Algorithm (Training — NLL Loss)

Used to compute the **partition function** — the log-sum of scores across ALL possible label sequences — which is then subtracted from the gold path score to compute the Negative Log-Likelihood loss.

```
Score(gold path) = start_score + Σ (transition + emission) + end_score
Partition        = log Σ_all_paths exp(score)
Loss             = -(Score(gold path) - Partition)
```

This ensures the model learns to assign high probability to correct entity sequences and low probability to invalid ones.

### 2. Viterbi Decoding (Inference — Optimal Path)

Finds the **globally optimal label sequence** using dynamic programming — avoiding the exponential brute-force search over all possible label paths.

```
Forward pass:  At each step t, track the best score to reach each label
               keeping backpointers to the best previous label

Backward pass: Starting from the best final label, trace back
               through the backpointers to reconstruct the full sequence
```

Time complexity: O(T × K²) where T = sequence length, K = number of labels — far more efficient than the O(K^T) brute force approach.

### 3. Sliding Window Inference

Handles medical texts longer than BERT's 512-token limit:

```
Long Text → Chunk 1 (tokens 0–512)
              stride=100
          → Chunk 2 (tokens 100–612)
              stride=100
          → Chunk 3 (tokens 200–712) ...

Results from all chunks → Deduplicated by (word, type) key → Final entity list
```

---

## 📁 Project Structure

```
Medical_Named_Entity_Recognition/
│
├── intelligent-medical-document-analyzer.ipynb   # Full training pipeline:
│                                                  #   Data loading (BC5CDR)
│                                                  #   BIO label encoding
│                                                  #   PubMedBERT + CRF fine-tuning
│                                                  #   Model evaluation (F1, Precision, Recall)
│                                                  #   Model export to HuggingFace Hub
│
├── app.py                                         # Streamlit web application:
│                                                  #   Model loading from HuggingFace Hub
│                                                  #   Sliding window inference pipeline
│                                                  #   Subword reassembly & entity extraction
│                                                  #   Interactive UI with entity display
│
└── requirements.txt                               # Python dependencies
```

---

## ▶ How to Run Locally

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Internet connection (model weights downloaded from HuggingFace Hub on first run)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/TirumalaRaoBoddana/Medical_Named_Entity_Recognition.git
cd Medical_Named_Entity_Recognition
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Application
```bash
streamlit run app.py
```

### Step 5 — Use the App
- Open your browser at `http://localhost:8501`
- Paste any clinical text or medical abstract into the text area
- Click **"Analyze Report"**
- View extracted **Diseases** (red panel) and **Chemicals/Drugs** (green panel) side by side

> **Note:** On first launch, the model (~440MB) is downloaded from HuggingFace Hub and cached automatically. Subsequent runs load from cache instantly via `@st.cache_resource`.

### Example Input
```
The patient was diagnosed with severe diabetic retinopathy and prescribed
metformin 500mg twice daily. A secondary finding of hypertension was noted,
and lisinopril was added to the treatment regimen.
```

### Expected Output
```
Diseases Found (2):          Chemicals/Drugs Found (2):
  - diabetic retinopathy       - metformin
  - hypertension               - lisinopril
```

---

## 🔄 Application Workflow

```
User pastes medical text into Streamlit UI
                ↓
Text tokenized with PubMedBERT tokenizer
(sliding window: max_length=512, stride=100)
                ↓
Each chunk passed through PubMedBERT encoder
  → 768-dimensional contextual token embeddings
                ↓
Linear classifier generates 5 emission scores per token
                ↓
Custom CRF Viterbi decoder finds optimal label path
  → Returns list of (token, BIO-label) pairs
                ↓
Post-processing:
  ├── Merge subword tokens (## prefix handling)
  ├── Group B-/I- spans into complete entity strings
  ├── Trim leading/trailing stop words
  └── Deduplicate across sliding window chunks
                ↓
Display:
  ├── 🔴 Diseases panel
  └── 🟢 Chemicals / Drugs panel
```

---

## 📊 Key Results & Metrics

| Metric | Value |
|--------|-------|
| **Dataset** | BC5CDR (BioCreative V CDR Benchmark) |
| **Base Model** | PubMedBERT (Microsoft BiomedNLP) |
| **F1 Score (Overall)** | 94% (improved from 92% baseline) |
| **Entity Types** | Disease, Chemical |
| **Label Schema** | BIO (5 labels) |
| **Max Input Length** | 512 tokens per chunk (sliding window for longer texts) |
| **Inference Mode** | Viterbi decoding (globally optimal sequence) |
| **Model Hosting** | HuggingFace Hub (`tirubujji92/pubmedbert-crf-ner-medical`) |
| **Deployment** | Streamlit web app with `@st.cache_resource` for fast repeated use |

---

## 💼 Skills Demonstrated

This project covers the full spectrum from research-level ML to production deployment:

- **Advanced NLP & Transformers:** Fine-tuned a domain-specific BERT model (PubMedBERT) on a biomedical NER benchmark, handling subword tokenization alignment for sequence labeling
- **Algorithm Implementation from Scratch:** Wrote a complete CRF layer in PyTorch including the Forward Algorithm (for training) and Viterbi Algorithm (for inference) — no external CRF libraries used
- **Sequence Labeling & Structured Prediction:** Applied BIO tagging conventions and learned transition matrices to enforce structurally valid entity label sequences
- **Long-Document Handling:** Designed a sliding window inference pipeline with overlapping strides to handle medical texts beyond the 512-token BERT limit
- **Model Hosting & MLOps:** Published trained model weights to HuggingFace Hub and implemented dynamic weight loading in the app using `hf_hub_download`
- **Python & PyTorch Engineering:** Architected a clean two-module class hierarchy (`PureCRF` + `PubMedBERT_CRF`) with proper separation of training and inference modes via forward pass branching
- **Software Development & Deployment:** Built and deployed a complete Streamlit application with caching, error handling, and a clean two-column UI layout
- **Version Control:** Full codebase managed and published on GitHub with reproducible `requirements.txt`

---

## 👤 Author

**Boddana Tirumala Rao**
B.Tech Computer Science Engineering | Batch 2026
Rajiv Gandhi University of Knowledge Technologies, Nuzvid

- GitHub: [@TirumalaRaoBoddana](https://github.com/TirumalaRaoBoddana)
- LinkedIn: [tirumala-rao-boddana](https://www.linkedin.com/in/tirumala-rao-boddana-9b5a6b274/)
- Email: btirumalarao27@gmail.com
- Model on HuggingFace: [tirubujji92/pubmedbert-crf-ner-medical](https://huggingface.co/tirubujji92/pubmedbert-crf-ner-medical)

---

> *This project was built as part of an academic and professional portfolio, demonstrating hands-on expertise in advanced NLP, deep learning architecture design, and end-to-end AI application development.*
