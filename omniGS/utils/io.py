# omniGS/utils/io.py
import os
import json
import pandas as pd
import numpy as np
import h5py
from typing import Any, Union
import logging

logger = logging.getLogger(__name__)


def read_csv(path: str, sep: str = ",") -> pd.DataFrame:
    if not os.path.exists(path):
        logger.error(f"CSV/TSV file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"Reading CSV/TSV file: {path}")
    return pd.read_csv(path, sep=sep)


def write_csv(df: pd.DataFrame, path: str, sep: str = ",", index: bool = False) -> None:
    df.to_csv(path, sep=sep, index=index)
    logger.info(f"Wrote CSV/TSV file: {path} (shape={df.shape}, sep='{sep}')")



def read_json(path: str, logger=None) -> Union[dict, list]:
    if logger is None:
        logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.error(f"JSON file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    logger.info(f"Reading JSON file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Any, path: str, indent: int = 2) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)



def write_hdf5(geno: np.ndarray, samples: list[str], path: str) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("geno", data=geno, compression="gzip")

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("samples", data=np.array(samples, dtype=dt))

    logger.info(f"Saved HDF5: {path} "
                f"(samples={len(samples)}, geno_shape={geno.shape})")


def read_hdf5(path: str) -> dict:
    if not os.path.exists(path):
        logger.error(f"HDF5 file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        geno = np.array(f["geno"])
        samples = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["samples"]]

    return {
        "geno": geno,
        "samples": samples
    }