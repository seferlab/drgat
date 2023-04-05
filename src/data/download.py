from __future__ import annotations

from pathlib import Path
import requests
from tqdm import tqdm

ZENODO_API = "https://zenodo.org/api/records/{record_id}"

def download_zenodo_record(record_id: int, out_dir: Path, force: bool = False) -> None:
    """Download all files from a Zenodo record.

    The paper states the data were downloaded from the Zenodo repository
    of Sharifi-Noghabi et al. (2019) (MOLI). The record id 4036592 is a
    public Zenodo dataset entry.

    This function downloads all 'files' listed by Zenodo into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = requests.get(ZENODO_API.format(record_id=record_id), timeout=60)
    meta.raise_for_status()
    j = meta.json()

    files = j.get("files", [])
    if not files:
        raise RuntimeError("No files found in Zenodo record. Check record_id.")

    for f in files:
        key = f["key"]
        url = f["links"]["self"]
        dest = out_dir / key
        if dest.exists() and not force:
            continue
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0"))
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {key}")
            with open(dest, "wb") as w:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        w.write(chunk)
                        pbar.update(len(chunk))
            pbar.close()
