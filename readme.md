# 🎓 Personalized Study Copilot (LLM + RAG)

An open-source **AI Study Copilot** that allows students to upload course materials (PDFs) and interact with them using a local Large Language Model.

The system uses **Retrieval-Augmented Generation (RAG)** to provide grounded answers based on uploaded study content instead of hallucinated responses.

---

## 🚀 Project Overview

This project builds a **fully local AI learning assistant** using:

* 📄 PDF document ingestion
* 🧠 Semantic embeddings
* 🔎 Vector similarity search (FAISS)
* 🤖 Local LLM inference (Mistral via Ollama)

The goal is to create a **personalized study assistant for graduate students** that works offline and preserves privacy.

---

## 🧠 Architecture

```
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embeddings (BGE Model)
      ↓
FAISS Vector Index
      ↓
Retriever
      ↓
Local LLM (Mistral)
      ↓
Grounded Answers
```

---

## ⚙️ Tech Stack

### LLM

* Mistral-7B (via Ollama)

### Embeddings

* `BAAI/bge-small-en-v1.5`
* Sentence Transformers

### Vector Database

* FAISS (CPU)

### Backend

* Python

### Tools

* VS Code
* Ollama
* NumPy

---

## 📂 Project Structure

```
study_copilot/
│
├── core/
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│
├── data/
│   └── sample.pdf
│
├── vectorstores/
├── database/
│
├── app.py
└── README.md
```


## 🔧 Installation

### 1️⃣ Clone Repository

```
git clone <repo-url>
cd study_copilot
```

---

### 2️⃣ Create Environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```
python -m pip install -r requirements.txt
```

(or install manually)

```
python -m pip install pypdf sentence-transformers faiss-cpu numpy
```

---

### 4️⃣ Install Ollama

Download from:

https://ollama.com

Pull model:

```
ollama pull mistral
```

---

## ▶️ Run Project

```
python app.py
```

---

## 🧩 How It Works

1. PDFs are converted into text.
2. Text is split into overlapping chunks.
3. Each chunk is converted into semantic embeddings.
4. Embeddings are stored in a FAISS index.
5. Future queries retrieve relevant chunks for LLM reasoning.

---

## 🎯 Project Goal

To build a **personalized AI study assistant** capable of:

* Context-aware Q&A
* Adaptive learning support
* Local and privacy-preserving inference

---

## 🚧 Upcoming Features

* Retriever module
* LLM integration
* Question answering interface
* Adaptive difficulty system
* Study analytics

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Shashank Kadiyala
Graduate Student (AI/ML) — Northeastern University (Fall 2025)

---
