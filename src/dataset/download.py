import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from tqdm.auto import tqdm

CODE15_URL = "https://zenodo.org/records/4916206/files"
CODE15_LABELS_URL = "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip"
SAMITROP_URL = "https://zenodo.org/records/4905618/files"
PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def fetch_and_extract(url, out_dir, expected, keep=None):
    """Extract a zip into out_dir, moving each expected file/folder to its top level."""
    out_dir = Path(out_dir)
    if all((out_dir / name).exists() for name in expected):
        return out_dir
    zip_path = fetch(url, out_dir / Path(urlparse(url).path).name)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if not m.startswith("__MACOSX") and (keep is None or keep(m))]
        zf.extractall(out_dir, members=members)
    zip_path.unlink()
    for name in expected:
        target = out_dir / name
        if not target.exists():
            found = min(out_dir.rglob(name), key=lambda p: len(p.parts))
            shutil.move(found, target)
    return out_dir


def download_code15(root, parts=(0,)):
    out = Path(root) / "code15"
    fetch(f"{CODE15_URL}/exams.csv?download=1", out / "exams.csv")
    fetch_and_extract(CODE15_LABELS_URL, out, ["code15_chagas_labels.csv"])
    for i in parts:
        fetch_and_extract(f"{CODE15_URL}/exams_part{i}.zip?download=1", out, [f"exams_part{i}.hdf5"])
    return out


def download_samitrop(root):
    out = Path(root) / "samitrop"
    fetch(f"{SAMITROP_URL}/exams.csv?download=1", out / "exams.csv")
    fetch_and_extract(f"{SAMITROP_URL}/exams.zip?download=1", out, ["exams.hdf5"])
    return out


def download_ptbxl(root):
    out = Path(root) / "ptbxl"
    # records100 is a downsampled copy of records500
    fetch_and_extract(PTBXL_URL, out, ["ptbxl_database.csv", "scp_statements.csv", "records500"],
                      keep=lambda m: "/records100/" not in m)
    return out
