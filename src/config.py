from pathlib import Path

from dotenv import load_dotenv
from os import getenv


load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")


BASE_DIR = Path(__file__).resolve().parent.parent

PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR = BASE_DIR / "temp_data"

