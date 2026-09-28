"""Download the open-weight models into HF_HOME, so containers
start without network access and without a first-request delay.

    HF_HOME=/app/hf python scripts/fetch_models.py
"""

from sentence_transformers import CrossEncoder, SentenceTransformer

from halden.config.settings import Settings

RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def main() -> None:
    settings = Settings()
    SentenceTransformer(settings.embedding_model, device="cpu")
    CrossEncoder(settings.reranker_model or RERANKER, device="cpu")
    print("models cached")


if __name__ == "__main__":
    main()
