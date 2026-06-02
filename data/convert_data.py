import math
import os

import awkward as ak
import h5py
import numpy as np
import uproot


# Read files with the correct format
def _read_parquet(filepath, branches, load_range=None):
    outputs = ak.from_parquet(filepath, columns=branches)
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
        outputs = outputs[start:stop]
    return outputs


def _read_hdf5_jetset(filepath, branches, load_range=None):
    """HDF5 reader for JetSet: structured arrays, index-based slicing."""
    with h5py.File(filepath, "r") as f:
        n_total = len(f[branches[0]])
        start, stop = (0, n_total) if load_range is None else load_range

        outputs = {}
        for k in branches:
            if k not in f:
                print(f"Warning: '{k}' not found in HDF5 file, skipping.")
                continue
            data = f[k][start:stop]
            for field in data.dtype.names:
                outputs[f"{k}_{field}"] = data[field]

    return ak.Array(outputs)


def _read_hdf5_atlastop(filepath, branches, load_range=None):
    """HDF5 reader for ATLASTopTagging: PyTables nodes, fraction-based slicing."""
    import tables

    tables.set_blosc_max_threads(4)

    with tables.open_file(filepath) as f:
        outputs = {}
        for k in branches:
            try:
                outputs[k] = getattr(f.root, k)[:]
            except tables.NoSuchNodeError:
                print(f"Warning: '{k}' not found in file, skipping.")

    if not outputs:
        return ak.Array({})

    n = len(next(iter(outputs.values())))
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * n)
    stop = max(start + 1, math.trunc(load_range[1] * n))

    return ak.Array({k: v[start:stop] for k, v in outputs.items()})


# Include fourmomenta as additional keys (in float64)
def _modify_jetset(inputs):
    """Compute track 4-momenta from pT / delta-eta / delta-phi."""
    pt = ak.values_astype(inputs["tracks_pt"], np.float64)
    deta = ak.values_astype(inputs["tracks_deta"], np.float64)
    dphi = ak.values_astype(inputs["tracks_dphi"], np.float64)
    jet_eta = ak.values_astype(inputs["jets_eta"], np.float64)
    jet_phi = ak.values_astype(inputs["jets_phi"], np.float64)

    tracks_eta = deta.to_numpy() + jet_eta.to_numpy().reshape(-1, 1)
    tracks_phi = dphi.to_numpy() + jet_phi.to_numpy().reshape(-1, 1)

    px = pt * np.cos(tracks_phi)
    py = pt * np.sin(tracks_phi)
    pz = pt * np.sinh(tracks_eta)

    inputs["tracks_px"] = px
    inputs["tracks_py"] = py
    inputs["tracks_pz"] = pz
    inputs["tracks_E"] = np.sqrt(px**2 + py**2 + pz**2)
    return inputs


def _modify_atlastop(inputs):
    """Compute cluster Cartesian momenta and split the binary label."""
    pt = ak.values_astype(inputs["fjet_clus_pt"], np.float64)
    eta = ak.values_astype(inputs["fjet_clus_eta"], np.float64)
    phi = ak.values_astype(inputs["fjet_clus_phi"], np.float64)

    inputs["fjet_clus_px"] = pt * np.cos(phi)
    inputs["fjet_clus_py"] = pt * np.sin(phi)
    inputs["fjet_clus_pz"] = pt * np.sinh(eta)

    inputs["label_isQCD"] = 1 - inputs["labels"]
    inputs["label_isTop"] = inputs["labels"]
    return inputs


# Configurations
DATASETS = {
    "jetset": dict(
        keys=["jets", "tracks", "truth_hadrons"],
        read_hdf5=_read_hdf5_jetset,
        modify=_modify_jetset,
        chunked=True,  # split large files into chunks
        chunk_size=100_000,
    ),
    "atlastop": dict(
        keys=[
            "EventInfo_mcEventNumber",
            "EventInfo_mcEventWeights",
            "fjet_clus_E",
            "fjet_clus_pt",
            "fjet_clus_eta",
            "fjet_clus_phi",
            "fjet_clus_taste",
            "labels",
            "fjet_C2",
            "fjet_D2",
            "fjet_ECF1",
            "fjet_ECF2",
            "fjet_ECF3",
            "fjet_L2",
            "fjet_L3",
            "fjet_Qw",
            "fjet_Split12",
            "fjet_Split23",
            "fjet_Tau1_wta",
            "fjet_Tau2_wta",
            "fjet_Tau3_wta",
            "fjet_Tau4_wta",
            "fjet_ThrustMaj",
            "fjet_eta",
            "fjet_phi",
            "fjet_pt",
            "fjet_m",
            "training_weights",
        ],
        read_hdf5=_read_hdf5_atlastop,
        modify=_modify_atlastop,
        chunked=False,
    ),
}


def convert(path_in, path_out, filename, dataset, input_type="hdf5"):
    cfg = DATASETS[dataset]
    keys = cfg["keys"]
    read_hdf5 = cfg["read_hdf5"]
    modify = cfg["modify"]
    chunked = cfg["chunked"]

    if input_type == "parquet":
        file_in = os.path.join(path_in, f"{filename}.parquet")
        _process_chunk(
            _read_parquet(file_in, keys), modify, os.path.join(path_out, f"{filename}.root")
        )
        return

    # HDF5 path
    file_in = os.path.join(path_in, f"{filename}.h5")

    if chunked:
        with h5py.File(file_in, "r") as f:
            n_total = len(f[keys[0]])
        print(f"Total jets: {n_total}")
        chunk_size = cfg["chunk_size"]
        ranges = range(0, n_total, chunk_size)
        print(f"Writing {len(ranges)} file(s) of up to {chunk_size} jets each.")

        for n, start in enumerate(ranges):
            outputs = read_hdf5(file_in, keys, load_range=(start, start + chunk_size))
            file_out = os.path.join(path_out, f"{filename}_{n:04d}.root")
            _process_chunk(outputs, modify, file_out)
    else:
        outputs = read_hdf5(file_in, keys)
        file_out = os.path.join(path_out, f"{filename}.root")
        _process_chunk(outputs, modify, file_out)


def _process_chunk(outputs, modify_fn, file_out):
    """Apply transformation, promote float16 → float32, and write to ROOT."""
    outputs = modify_fn(outputs)

    arrays = {}
    for k in outputs.fields:
        arr = outputs[k]
        if ak.to_numpy(arr).dtype == np.float16:
            arr = ak.values_astype(arr, np.float32)
        arrays[k] = arr

    with uproot.recreate(file_out, compression=uproot.LZ4(4)) as f:
        f["Events"] = arrays
    print(f"  Finished with {file_out}")


if __name__ == "__main__":
    # JetSet
    jetset_cfg = dict(
        path_in="/globalsc/ucl/cp3/favaro/JetSet/",
        path_out="/globalsc/ucl/cp3/favaro/JetSet/roots/",
        dataset="jetset",
        input_type="hdf5",
        folders=["small/", "medium/", "large/"],
        files=["mc-flavtag-ttbar-small", "mc-flavtag-ttbar-medium", "mc-flavtag-ttbar-large"],
    )

    # ATLASTop
    atlastop_cfg = dict(
        path_in="/globalsc/users/f/a/favaro/ATLASTopTagging/h5s/",
        path_out="/globalsc/users/f/a/favaro/ATLASTopTagging/roots_f64/",
        dataset="atlastop",
        input_type="hdf5",
        folders_ranges=[
            ("train_nominal", range(0, 929)),
            ("test_nominal", range(0, 104)),
            ("angular", range(0, 50)),
            ("bias", range(0, 101)),
            ("cer", range(0, 101)),
            ("cluster", range(0, 51)),
            ("cpos", range(0, 101)),
            ("dipole", range(0, 50)),
            ("esdown", range(0, 101)),
            ("esup", range(0, 101)),
            ("string", range(0, 51)),
            ("teg", range(0, 75)),
            ("tej", range(0, 71)),
            ("tfj", range(0, 70)),
            ("tfl", range(0, 60)),
            ("ttbar_pythia", range(0, 2)),
            ("ttbar_herwig", range(0, 2)),
        ],
    )

    # JetSet conversion
    for folder, fname in zip(jetset_cfg["folders"], jetset_cfg["files"], strict=True):
        p_in = os.path.join(jetset_cfg["path_in"], folder)
        p_out = os.path.join(jetset_cfg["path_out"], folder)
        os.makedirs(p_out, exist_ok=True)
        print(f"\n[JetSet] Converting {folder} to {p_out}")
        convert(p_in, p_out, fname, jetset_cfg["dataset"], jetset_cfg["input_type"])

    # ATLASTop conversion
    for folder, filerange in atlastop_cfg["folders_ranges"]:
        p_in = os.path.join(atlastop_cfg["path_in"], folder)
        p_out = os.path.join(atlastop_cfg["path_out"], folder)
        os.makedirs(p_out, exist_ok=True)
        print(f"\n[ATLASTop] Converting {folder} to {p_out}")
        for idx in filerange:
            fname = f"{folder}_{idx:03d}"
            convert(p_in, p_out, fname, atlastop_cfg["dataset"], atlastop_cfg["input_type"])
