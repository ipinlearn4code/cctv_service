import pandas as pd
import filelock
import os

def read_csv(file_path: str, columns: list):
    lock = filelock.FileLock(f"{file_path}.lock")
    with lock:
        if not os.path.exists(file_path):
            return pd.DataFrame(columns=columns)
        return pd.read_csv(file_path)

def write_csv(file_path: str, df: pd.DataFrame):
    lock = filelock.FileLock(f"{file_path}.lock")
    with lock:
        df.to_csv(file_path, index=False)
