from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
# DATA_PATH = root / "data/"  # datasets and pretrained weights
DATA_PATH = Path("/is/cluster/fast/hli/glue/data/")
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
