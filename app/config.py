from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tech_stacks.csv"

DEFAULT_TOP_N = 3
MIN_USER_INPUTS = 3
