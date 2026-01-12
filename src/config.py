from pathlib import Path

from dotenv import load_dotenv
from os import getenv


load_dotenv()
MODEL_ID = getenv("MODEL_ID")


BASE_DIR = Path(__file__).resolve().parent.parent

PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR = BASE_DIR / "temp_data"


PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

