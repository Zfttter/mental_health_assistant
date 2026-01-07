import os
import pandas as pd
import pickle
import minsearch
import logging

_DEFAULT_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "data.csv"))
DATA_PATH = os.getenv("DATA_PATH", _DEFAULT_DATA_PATH)

def load_index(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    documents = df.to_dict(orient="records")
    logging.getLogger(__name__).info("Loaded %d documents for indexing", len(documents))

    index = minsearch.Index(
        text_fields=['Questions', 'Answers'],
        keyword_fields=["Question_ID"],
    )

    index.fit(documents)
    return index
