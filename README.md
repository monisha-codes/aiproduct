# ⚖️ Legal RAG System — Query Processing Modules (M-04, M-05, M-06)

## 📌 Overview

This project implements the core query processing layer of a Legal RAG (Retrieval-Augmented Generation) system.  
It focuses on validating, cleaning, and understanding user queries before retrieval.

Modules included:
- M-04: Query Validation Gate
- M-05: Query Preprocessing
- M-06: Intent & Domain Classification

---

## 🧠 Architecture Flow

User Query
   ↓
M-04 → Validation + PII Masking
   ↓
M-05 → Cleaning + Expansion
   ↓
M-06 → Intent + Classification
   ↓
Output → Retrieval Engine

---

## 🚀 Features

- PII detection and masking (US-based)
- Legal domain validation
- Query cleaning and normalization
- Legal abbreviation expansion (USC, ADA, etc.)
- Basic entity extraction
- Intent and domain classification (zero-shot model)
- Structured logging (JSON)
- Error handling and fallback mechanisms
- FastAPI-based API service
- Modular and scalable design

---

## 🛠️ Tech Stack

- Python 3.10
- FastAPI
- spaCy
- Transformers (HuggingFace)
- PyTorch
- Langdetect

---

## 📁 Project Structure

app/
│
├── main.py
├── config.py
├── logger.py
│
├── api/
│   └── routes.py
│
├── models/
│   └── schemas.py
│
├── services/
│   ├── validation_service.py
│   ├── preprocessing_service.py
│   ├── classification_service.py
│
├── utils/
│   └── pii.py
│
├── requirements.txt
├── Dockerfile

---

## ⚙️ Setup Instructions

1. Clone repository

git clone <repo-url>
cd legal-rag-system

2. Create virtual environment

python -m venv venv

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Download spaCy model

python -m spacy download en_core_web_sm

---

## ▶️ Run the Application

uvicorn app.main:app --reload

---

## 🌐 API Access

http://127.0.0.1:8000/docs

---

## 📡 API Endpoint

POST /v1/process

### Request

{
  "query": "Explain Section 1983 USC for John Doe 9876543210"
}

---

### Response

{
  "status": "success",
  "data": {
    "validation": {
      "query": "Explain Section 1983 USC for [NAME] [PHONE]",
      "pii_masked": true
    },
    "preprocessing": {
      "expanded_query": "Explain Section 1983 United States Code..."
    },
    "classification": {
      "domain": "constitutional law",
      "intent": "definition",
      "jurisdiction": ["US"]
    }
  }
}

---

## 🔐 Security

- No raw PII stored or logged
- PII masked before processing
- Input validation enforced

---

## ⚡ Performance

- Stateless API design
- Lazy model loading
- Scalable for large workloads
- Ready for caching integration

---

## 🧪 Testing

- Unit tests recommended for:
  - Validation logic
  - PII masking
  - Classification

---

## 🐳 Docker

Build:

docker build -t legal-rag .

Run:

docker run -p 8000:8000 legal-rag

---

## 🎯 Summary

This module converts raw user queries into structured, validated, and classified inputs to enable accurate legal information retrieval.