from pathlib import Path
from urllib.request import urlopen

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DEST = Path(__file__).with_name("data") / "winequality-red.csv"
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_bytes(urlopen(URL, timeout=45).read())
print(f"Saved dataset to {DEST}")
