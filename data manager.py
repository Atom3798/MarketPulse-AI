import pandas as pd
import numpy as np

class DataManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def engineer_features(self) -> pd.DataFrame:
        df = self.df.copy()
        df["return"] = df["price"].pct_change()
        df["ma5"] = df["price"].rolling(5).mean()
        df["volatility"] = df["return"].rolling(10).std()
        df = df.dropna()
        return df
