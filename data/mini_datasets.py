from pathlib import Path

import uproot

"""
Create 'mini' datasets from standard files
Works for JetClass, TopTagXL, ATLASTop, and JetSet
How to use this file to create the mini version of a full dataset:
- create the folder "original/" for the dataset, e.g. jetclass/original/
- create the folder structure of the original dataset in "original/", e.g. train, test, val
- copy the first file per class, or label, to the corresponding folder
- run "python3 mini_datasets.py" after adjusting the list of labels in the main function, if needed

The script creates the the correct folder structure within e.g. jetclass/
and a single file per class/label with the specified number of events.
"""


def get_properties(label):
    if label == "jetclass":
        in_dir = "jetclass"
        out_dir = "jetclass"
        labels = ["train_100M", "test_20M", "val_5M"]
        nevents = 1000
    elif label == "toptagxl":
        in_dir = "toptagxl"
        out_dir = "toptagxl"
        labels = ["train_100M", "test_25M", "val_10M"]
        nevents = 5000
    elif label == "atlastop":
        in_dir = "atlastop"
        out_dir = "atlastop"
        labels = [
            "train_nominal",
            "val_nominal",
            "test_nominal",
            "ttbar_herwig",
            "ttbar_pythia",
        ]
        nevents = 5000
    elif label == "jetset":
        in_dir = "jetset"
        out_dir = "jetset"
        labels = ["large", "medium", "small"]
        nevents = 5000
    return in_dir, out_dir, labels, nevents


def recreate(dataset, out_dir, labels, nevents):
    in_dir = Path(f"{dataset}/files_original")
    for label in labels:
        input_file_path = in_dir / label
        file_paths = [p for p in input_file_path.iterdir() if p.is_file()]
        print(file_paths)
        for path in file_paths:
            with uproot.open(path) as fin:
                key = fin.keys()[0]
                tree = fin[key]
                print(key, fin, tree)

                # Exclude explicit count branches; uproot will recreate them on write
                # This only affects TopTagXL
                if dataset == "toptagxl":
                    counter_names = {
                        branch.count_branch.name
                        for _, branch in tree.items()
                        if branch.count_branch is not None
                    }
                else:
                    counter_names = {}
                branch_names = [name for name in tree.keys() if name not in counter_names]

                data = tree.arrays(
                    branch_names,
                    library="ak",
                    how=dict,
                    entry_start=0,
                    entry_stop=nevents,
                )

                prefix, n = path.stem.rsplit("_", 1)
                if dataset == "jetset":
                    name = f"{prefix}_{int(n):04d}{path.suffix}"
                else:
                    name = f"{prefix}_{int(n):03d}{path.suffix}"
                outfile = out_dir / label / name
                outfile.parent.mkdir(parents=True, exist_ok=True)
                with uproot.recreate(outfile, compression=uproot.LZ4(4)) as fout:
                    fout["tree"] = data
                print(f"{path} -> {outfile}")


if __name__ == "__main__":
    for label in ["jetclass", "toptagxl", "atlastop", "jetset"]:
        props = get_properties(label)
        recreate(*props)
