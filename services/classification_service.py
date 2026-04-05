from transformers import pipeline
from config import settings

classifier = None

def get_model():
    global classifier
    if classifier is None:
        classifier = pipeline("zero-shot-classification", model=settings.MODEL_NAME)
    return classifier


DOMAINS = ["contract law", "criminal law", "tax law"]
INTENTS = ["definition", "procedure", "case lookup"]


def classify_query(data):
    try:
        model = get_model()
        query = data["expanded_query"]

        domain = model(query, DOMAINS)["labels"][0]
        intent = model(query, INTENTS)["labels"][0]

        return {
            "domain": domain,
            "intent": intent,
            "jurisdiction": ["US"],
            "complexity": "simple"
        }

    except Exception as e:
        # fallback
        return {
            "domain": "unknown",
            "intent": "unknown",
            "complexity": "simple",
            "fallback": True
        }