from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("models") / "all-MiniLM-L6-v2"

def main():
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(OUT_DIR))
    print(f"Saved model to: {OUT_DIR}")

if __name__ == "__main__":
    main()