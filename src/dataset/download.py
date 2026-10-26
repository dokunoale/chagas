import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import gdown as gd
from tqdm.auto import tqdm

CODE15_URL = "https://zenodo.org/records/4916206/files"
CODE15_LABELS_URL = "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip"
SAMITROP_URL = "https://zenodo.org/records/4905618/files"


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


def extract(zip_path, out_dir, expected):
    """Extract a zip into out_dir, moving each expected file to its top level."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir, members=[m for m in zf.namelist() if not m.startswith("__MACOSX")])
    Path(zip_path).unlink()
    for name in expected:
        target = out_dir / name
        if not target.exists():
            found = min(out_dir.rglob(name), key=lambda p: len(p.parts))
            shutil.move(found, target)
    for d in sorted(out_dir.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def fetch_and_extract(url, out_dir, expected):
    out_dir = Path(out_dir)
    if all((out_dir / name).exists() for name in expected):
        return out_dir
    extract(fetch(url, out_dir / Path(urlparse(url).path).name), out_dir, expected)
    return out_dir


def fetch_from_drive(folder, out_dir, names):
    """Download the wanted files from a shared Drive folder (link or id).

    The folder holds the original files: each name can be there as is or zipped (`<stem>.zip`).
    """
    out_dir = Path(out_dir)
    missing = [name for name in names if not (out_dir / name).exists()]
    if not missing:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    key = "url" if folder.startswith("http") else "id"
    listing = gd.download_folder(**{key: folder}, skip_download=True, quiet=True)
    files = {Path(f.path).name: f.id for f in listing}
    for name in missing:
        zipped = Path(name).with_suffix(".zip").name
        if name in files:
            gd.download(id=files[name], output=str(out_dir / name))
        elif zipped in files:
            gd.download(id=files[zipped], output=str(out_dir / zipped))
            extract(out_dir / zipped, out_dir, [name])
        else:
            raise FileNotFoundError(f"neither {name} nor {zipped} in the Drive folder")
    return out_dir


def download_code15(root, parts=(0,), gdown=None):
    """Each part is a single HDF5 of ~20k exams (2.7 GB zipped), the smallest downloadable unit."""
    out = Path(root) / "code15"
    if gdown:
        return fetch_from_drive(gdown, out, ["exams.csv", "code15_chagas_labels.csv"]
                                + [f"exams_part{i}.hdf5" for i in parts])
    fetch(f"{CODE15_URL}/exams.csv?download=1", out / "exams.csv")
    fetch_and_extract(CODE15_LABELS_URL, out, ["code15_chagas_labels.csv"])
    for i in parts:
        fetch_and_extract(f"{CODE15_URL}/exams_part{i}.zip?download=1", out, [f"exams_part{i}.hdf5"])
    return out


def download_samitrop(root, gdown=None):
    out = Path(root) / "samitrop"
    if gdown:
        return fetch_from_drive(gdown, out, ["exams.csv", "exams.hdf5"])
    fetch(f"{SAMITROP_URL}/exams.csv?download=1", out / "exams.csv")
    fetch_and_extract(f"{SAMITROP_URL}/exams.zip?download=1", out, ["exams.hdf5"])
    return out
