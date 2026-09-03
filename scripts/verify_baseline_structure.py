"""Read-only structural verification; it never trains or writes baseline artifacts."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data" / "26" / "results" / "best_hybrid.pt"


def main():
    print("Original source:", (ROOT / "26_cognitive_affective_multimodal_support.py").exists())
    print("Frozen checkpoint:", CHECKPOINT)
    print("Checkpoint exists:", CHECKPOINT.exists())
    print("No training or checkpoint writes are performed by this verifier.")


if __name__ == "__main__":
    main()
