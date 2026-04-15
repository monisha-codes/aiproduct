# ⚖️ Legal RAG API – Intelligent Legal Query Processing System

This project implements a **modular Legal RAG (Retrieval-Augmented Generation) pipeline** designed to process, understand, and classify legal queries with high accuracy and safety.

It combines:
- Rule-based NLP
- ML models (Legal-BERT / classifiers)
- LLMs (via Ollama)
- Domain-specific preprocessing

---

# 🧩 System Architecture

```text
User Query
   ↓
Validation Module
   ↓
Preprocessing Module
   ↓
Classification Module
   ↓
(RAG / Retrieval / Future Response Layer)
🔹 Core Modules
1️⃣ Validation Module
📌 Purpose

Ensures input is:

Clean
Safe
Legal-domain relevant
Free from sensitive information
⚙️ Responsibilities
Query normalization
Language detection
Token counting
Domain classification (legal vs non-legal)
PII masking:
Name
Email
Phone
Confidence scoring
🤖 Models Used
Component	Technology
Domain Classifier	ML model (custom / sklearn / lightweight classifier)
Legal Context	Legal-BERT
Fallback	LLM (via Ollama)
📥 Input
{
  "query": "Help John Doe john@example.com 9876543210 with tax filing rules"
}
📤 Output
{
  "query": "help [NAME] doe [EMAIL] [PHONE] with tax filing rules",
  "pii_masked": true,
  "token_count": 9,
  "lang": "en",
  "domain_confidence": 0.91
}
2️⃣ Preprocessing Module
📌 Purpose

Transforms queries into structured, enriched, retrieval-ready format

⚙️ Responsibilities
Cleaning and normalization
Abbreviation expansion
Entity extraction
Query restructuring
LLM-based rewriting (fallback only)
🧠 Key Features
Preserves meaningful queries
Avoids over-rewriting
Handles short / ambiguous queries
Keeps placeholders intact:
[NAME]
[EMAIL]
[PHONE]
🤖 Models Used
Component	Technology
Abbreviation Expansion	Embedding model
Entity Extraction	Rule-based + NLP
Query Rewriting	Ollama LLM
Embeddings	Sentence Transformers / custom embedder
📥 Input
{
  "query": "hipaa"
}
📤 Output
{
  "original_query": "hipaa",
  "cleaned_query": "HIPAA",
  "expanded_query": "Health Insurance Portability and Accountability Act (HIPAA)",
  "restructured_query": "Explain the legal meaning of Health Insurance Portability and Accountability Act (HIPAA).",
  "abbreviations": {
    "HIPAA": "Health Insurance Portability and Accountability Act"
  },
  "entities": {
    "acts": ["Health Insurance Portability and Accountability Act"],
    "sections": [],
    "citations": [],
    "courts": []
  }
}
3️⃣ Classification Module
📌 Purpose

Identifies:

Legal domain
User intent
Jurisdiction
⚙️ Responsibilities
Domain classification:
Civil law
Criminal law
Corporate law
Compliance
Intent detection:
Definition
Explanation
Compliance
Case lookup
Legal procedure
Jurisdiction detection:
US
India
Extendable
🤖 Models Used
Component	Technology
Primary Classifier	ML model
Semantic Understanding	Legal-BERT
Fallback	Ollama LLM
📥 Input
{
  "restructured_query": "Explain HIPAA rules"
}
📤 Output
{
  "domain": "civil law",
  "intent": "compliance and regulation",
  "jurisdiction": ["US"]
}
🤖 LLM Integration (Ollama)

The system uses local LLMs via Ollama for:

Query rewriting
Validation fallback
Classification fallback
System Q&A (capabilities)
🔧 Supported Models
Model	Size	Usage
qwen2.5:3b	~2GB	Recommended (fast + lightweight)
phi3	~2.2GB	Alternative
llama3	~4.7GB	Higher quality
🧪 Example Config
{
  "model": "qwen2.5:3b",
  "base_url": "http://127.0.0.1:11434/api/generate"
}
🛠️ Tech Stack
Backend
FastAPI
Python 3.10+
NLP / ML
Legal-BERT
Sentence Transformers
Custom classifiers
LLM
Ollama (local inference)
Qwen / Phi / LLaMA models
Utilities
Regex-based parsing
Abbreviation store
Embedding search
🔐 Safety Features
✅ PII masking by default
✅ Fail-safe fallback (LLM errors won't break pipeline)
✅ Domain filtering (blocks non-legal queries)
✅ Minimal query distortion
⚡ Example End-to-End
Input
{
  "query": "Explain HIPAA rules"
}
Output
{
  "validation": {...},
  "preprocessing": {...},
  "classification": {
    "domain": "civil law",
    "intent": "compliance and regulation",
    "jurisdiction": ["US"]
  }
}
🚀 Future Enhancements
🔍 Vector database integration (FAISS / Pinecone)
📚 Legal document ingestion pipeline
🌍 Multi-language support
🧠 Multi-turn conversation memory
⚖️ Case law retrieval
📄 PDF / DOCX parsing
🧪 Running the Project
uvicorn main:app --reload

Swagger UI:

http://127.0.0.1:8000/docs
👩‍💻 Project Vision

This system is designed for:

Legal AI assistants
Compliance automation
Contract analysis tools
Enterprise legal intelligence systems
Large-scale legal document processing
📌 Key Design Principles
Modular architecture
Rule-first, LLM-second approach
High reliability
Low latency (local LLMs)
Production-ready scalability
