import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import wfdb

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SOURCES = ["CODE-15%", "SaMi-Trop", "PTB-XL"]
FS = {"CODE-15%": 400, "SaMi-Trop": 400, "PTB-XL": 500}

# Shared schema; source-specific columns follow and are NaN elsewhere.
# `file` + `index` locate the raw signal (index is the row in the HDF5, -1 for WFDB).
COLUMNS = ["record_id", "patient_id", "source", "chagas", "age", "is_male", "fs", "file", "index"]

CODE15_FLAGS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg"]
PTBXL_EXTRAS = ["scp_codes", "strat_fold", "device", "baseline_drift", "static_noise",
                "burst_noise", "electrodes_problems", "extra_beats", "pacemaker"]


def _to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "t", "yes"])


def load_code15(root, parts=(0,)):
    root = Path(root) / "code15"
    exams = pd.read_csv(root / "exams.csv")
    labels = pd.read_csv(root / "code15_chagas_labels.csv")[["exam_id", "chagas"]]
    index = []
    for i in parts:
        path = root / f"exams_part{i}.hdf5"
        with h5py.File(path, "r") as f:
            ids = f["exam_id"][:]
        index.append(pd.DataFrame({"exam_id": ids, "file": str(path), "index": np.arange(len(ids))}))
    meta = pd.concat(index).merge(exams, on="exam_id").merge(labels, on="exam_id")
    meta = meta.assign(
        record_id="code15_" + meta.exam_id.astype(str),
        patient_id="code15_" + meta.patient_id.astype(str),
        source="CODE-15%",
        chagas=_to_bool(meta.chagas),
        is_male=_to_bool(meta.is_male),
        fs=FS["CODE-15%"],
    )
    return meta[COLUMNS + CODE15_FLAGS + ["nn_predicted_age"]].reset_index(drop=True)


def load_samitrop(root):
    root = Path(root) / "samitrop"
    exams = pd.read_csv(root / "exams.csv")
    with h5py.File(root / "exams.hdf5", "r") as f:
        assert len(f["tracings"]) == len(exams), "exams.csv and exams.hdf5 are not aligned"
    meta = exams.assign(
        record_id="samitrop_" + exams.exam_id.astype(str),
        patient_id="samitrop_" + exams.exam_id.astype(str),  # first exam only: one per patient
        source="SaMi-Trop",
        chagas=True,
        is_male=_to_bool(exams.is_male),
        fs=FS["SaMi-Trop"],
        file=str(root / "exams.hdf5"),
        index=np.arange(len(exams)),
    )
    meta["normal_ecg"] = _to_bool(exams.normal_ecg)
    return meta[COLUMNS + ["normal_ecg"]]


def load_ptbxl(root):
    root = Path(root) / "ptbxl"
    db = pd.read_csv(root / "ptbxl_database.csv")
    meta = db.assign(
        record_id="ptbxl_" + db.ecg_id.astype(str),
        patient_id="ptbxl_" + db.patient_id.astype(int).astype(str),
        source="PTB-XL",
        chagas=False,
        is_male=db.sex == 0,
        fs=FS["PTB-XL"],
        file=[str(root / f) for f in db.filename_hr],
        index=-1,
        scp_codes=db.scp_codes.apply(ast.literal_eval),
    )
    return meta[COLUMNS + PTBXL_EXTRAS]


def load_all(root, code15_parts=(0,)):
    meta = pd.concat([load_code15(root, code15_parts), load_samitrop(root), load_ptbxl(root)],
                     ignore_index=True)
    meta["chagas"] = meta.chagas.astype(bool)
    meta["is_male"] = meta.is_male.astype(bool)
    return meta


def read_signals(meta):
    """Raw signals in mV, each of shape (n_samples, 12), in the row order of `meta`."""
    meta = meta.reset_index(drop=True)
    out = [None] * len(meta)
    hdf5 = meta["index"] >= 0
    for file, g in meta[hdf5].groupby("file", sort=False):
        order = np.argsort(g["index"].to_numpy())
        idx = g["index"].to_numpy()[order]
        # A contiguous slice is much faster than h5py point selection
        sel = slice(idx[0], idx[-1] + 1) if idx[-1] - idx[0] + 1 == len(idx) else idx
        with h5py.File(file, "r") as f:
            x = f["tracings"][sel]
        for pos, sig in zip(g.index[order], x):
            out[pos] = sig.astype(np.float32)
    for pos, file in meta.loc[~hdf5, "file"].items():
        sig, fields = wfdb.rdsamp(file)
        assert [n.upper() for n in fields["sig_name"]] == LEADS, file
        out[pos] = sig.astype(np.float32)
    return out


def iter_signals(meta, chunk_size=1000):
    for start in range(0, len(meta), chunk_size):
        chunk = meta.iloc[start:start + chunk_size]
        yield chunk, read_signals(chunk)
