from utils.abbreviation_store import extract_abbreviations

def process_document(text: str):
    try:
        # ✅ STEP 1: Learn abbreviations
        extract_abbreviations(text)

        # (Optional future steps)
        # chunk_text(text)
        # store_embeddings(text)

        return {"status": "processed"}

    except Exception as e:
        return {"error": str(e)}