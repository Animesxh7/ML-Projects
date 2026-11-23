"""Download and extract the UCI Bank Marketing dataset."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT = Path(__file__).with_name("data.csv")

print("Downloading UCI Bank Marketing dataset...")
with urlopen(URL, timeout=60) as response:
    outer_bytes = response.read()

with ZipFile(BytesIO(outer_bytes)) as outer:
    inner_name = next((n for n in outer.namelist() if n.lower().endswith("bank.zip")), None)
    if inner_name is None:
        raise RuntimeError("Could not find bank.zip inside the UCI archive.")
    inner_bytes = outer.read(inner_name)

with ZipFile(BytesIO(inner_bytes)) as inner:
    csv_name = next((n for n in inner.namelist() if n.lower().endswith("bank-full.csv")), None)
    if csv_name is None:
        raise RuntimeError("Could not find bank-full.csv inside bank.zip.")
    OUT.write_bytes(inner.read(csv_name))

print(f"Saved {OUT.name}. Run the Streamlit app with: streamlit run app.py")
