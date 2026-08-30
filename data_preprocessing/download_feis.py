"""Download and extract FEIS (Wellington & Clayton, Zenodo 2019).

Source: https://zenodo.org/records/3554128
21 English and 2 Chinese participants, about 1.6 GB.
Open Data Commons Attribution License.
"""

import zipfile

import requests
from tqdm import tqdm

from paths import RAW_ROOT

URL = ("https://zenodo.org/api/records/3554128/files/"
       "scottwellington/FEIS-v1.1.zip/content")
RAW_DIR = RAW_ROOT / "feis"
ARCHIVE = RAW_DIR / "FEIS-v1.1.zip"


def download_file(url, dest):
    if dest.exists():
        print(f"skip download, exists: {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract_archive(archive, dest_dir):
    marker = dest_dir / ".extracted"
    if marker.exists():
        print(f"skip extract, exists: {dest_dir.name}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    marker.touch()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_file(URL, ARCHIVE)
    extract_archive(ARCHIVE, RAW_DIR)


if __name__ == "__main__":
    main()
