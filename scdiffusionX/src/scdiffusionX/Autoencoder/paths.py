from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.resolve()

DATA_DIR = ROOT / "training_script" / "training_autoencoder" / "outputs" / "datasets"
TRAINING_FOLDER = ROOT / "training_script" / "training_autoencoder" / "outputs" / "checkpoints"
