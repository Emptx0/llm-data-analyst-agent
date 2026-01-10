from dataclasses import dataclass
import pandas as pd


@dataclass
class DataContext:
    df: pd.DataFrame | None = None
    path: str | None = None
    format: str | None = None

    def is_loaded(self) -> bool:
        return self.df is not None


DATA_CONTEXT = DataContext()

