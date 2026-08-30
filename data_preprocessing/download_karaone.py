"""Download and extract KaraOne (Zhao & Rudzicz, ICASSP 2015).

Source: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html
14 subjects, about 24 GB. Free for academic, non-profit use.
"""

import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from paths import RAW_ROOT

BASE_URL = "http://www.cs.toronto.edu/~complingweb/data/karaOne"
SUBJECTS = [
    "MM05", "MM08", "MM09", "MM10", "MM11", "MM12", "MM14",
    "MM15", "MM16", "MM18", "MM19", "MM20", "MM21", "P02",
]
RAW_DIR = RAW_ROOT / "karaone"


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
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(dest_dir)
    marker.touch()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        archive = RAW_DIR / f"{subject}.tar.bz2"
        download_file(f"{BASE_URL}/{subject}.tar.bz2", archive)
        extract_archive(archive, RAW_DIR / subject)


if __name__ == "__main__":
    main()
