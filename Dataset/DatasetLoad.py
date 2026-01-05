import os
from pathlib import Path
import pandas as pd

DATA_ROOT = Path(os.environ["WINTERLAB_VIDEO_ROOT"])
SPLIT_ROOT = Path(os.environ["WINTERLAB_SPLIT_ROOT"])

#example using train data

df = pd.read_csv(SPLIT_ROOT / "train.csv")

for i in range(len(df)):
    full_path = DATA_ROOT / df.loc[i, "path"]
    print(full_path)
