SYSTEM_QA = {
    "capabilities": {
        "patterns": [
            "what can you do",
            "your capabilities",
            "how can you help",
            "what do you support"
        ],
        "answer": (
            "I am a Legal AI system designed to process large legal documents. "
            "I can validate legal queries, mask sensitive information, correct spelling errors, "
            "expand legal abbreviations, restructure queries for better understanding, "
            "classify legal domain and intent, and assist retrieval-based legal analysis."
        )
    },

    "pipeline": {
        "patterns": [
            "how does your system work",
            "workflow",
            "pipeline",
            "how do you process queries"
        ],
        "answer": (
            "My system follows a structured pipeline: "
            "validation to ensure the query is legal and safe, "
            "preprocessing to clean, correct, and restructure the query, "
            "and classification to determine domain and intent. "
            "This improves accuracy for large-scale legal document processing."
        )
    },

    "preprocessing": {
        "patterns": [
            "how do you preprocess",
            "what happens in preprocessing",
            "query cleaning"
        ],
        "answer": (
            "In preprocessing, I clean the query, correct spelling mistakes, "
            "expand legal abbreviations, detect entities, and restructure the query "
            "into a more meaningful legal form."
        )
    },

    "pii": {
        "patterns": [
            "pii",
            "sensitive data",
            "mask data",
            "privacy detection"
        ],
        "answer": (
            "I detect and mask sensitive information such as names, phone numbers, "
            "email addresses, and social security numbers to ensure privacy compliance."
        )
    },

    "abbreviation": {
        "patterns": [
            "abbreviation",
            "hipaa",
            "gdpr",
            "legal short forms"
        ],
        "answer": (
            "I can expand legal abbreviations such as HIPAA, GDPR, USC, and CFR. "
            "I also learn new abbreviations dynamically from ingested legal documents."
        )
    },

    "classification": {
        "patterns": [
            "classification",
            "domain detection",
            "intent detection"
        ],
        "answer": (
            "I classify queries based on legal domain, intent, and jurisdiction using "
            "a hybrid approach combining rules, machine learning, and LLM-based refinement."
        )
    },

    "documents": {
        "patterns": [
            "legal documents",
            "upload documents",
            "process documents",
            "ingestion"
        ],
        "answer": (
            "I can process large legal documents to extract abbreviations and prepare them "
            "for retrieval-based workflows such as legal search and analysis."
        )
    },

    "rag": {
        "patterns": [
            "rag",
            "retrieval",
            "context retrieval",
            "document search"
        ],
        "answer": (
            "I support Retrieval-Augmented Generation (RAG), where I use relevant legal "
            "documents to improve query understanding and provide context-aware responses."
        )
    },

    "models": {
        "patterns": [
            "what models do you use",
            "llm",
            "which model",
            "ai model"
        ],
        "answer": (
            "I use a combination of models including zero-shot classifiers, "
            "sentence transformers for semantic understanding, and LLMs such as Ollama "
            "for query rewriting and fallback reasoning."
        )
    },

    "limitations": {
        "patterns": [
            "limitations",
            "what can't you do",
            "accuracy"
        ],
        "answer": (
            "While I provide strong legal query processing, I do not replace legal advice. "
            "My responses depend on available data and models, and complex legal interpretation "
            "should be reviewed by professionals."
        )
    }
}