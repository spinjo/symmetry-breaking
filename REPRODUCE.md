
## Reproducing the results

### 1) Setup environment

```bash
git clone https://github.com/spinjo/symmetry-breaking
cd symmetry-breaking
```

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install -r requirements_nodeps.txt --no-deps
```
Note that `flash-attn` and `xformers` might require separate installations. The `--no-deps` requirements install core components for the PET and Salt transformers.

The repo already contains 'mini' versions of all datasets in the `data/` folder, allowing to run everything without downloading big datasets. Use the (default) `config_quick` folder to use them. Typical commands are

```bash
python run.py -cn toptagging save=false
python run.py -cn jetclass save=false
python run.py -cn toptagxl save=false
python run.py -cn atlastop save=false
python run.py -cn jetset save=false
```

The code supports multi-GPU and multi-node runs using `torchrun`. We currently do not use this widely, but it should be correctly implemented. For instance, the syntax for running on 1 node with 4 GPUs, or 2 nodes with 4 GPUs each is

```bash
# single-node
torchrun --nproc-per-node=4 run.py save=false

# multi-node (using slurm)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
srun --cpu-bind=none torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=gpu \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID \
  run.py save=false
```

### 2) Collect datasets

To download the full datasets, do this:

- Top-tagging: `python data/collect_data.py toptagging`
- JetClass: Download 180GB dataset from https://zenodo.org/records/6619768; update the path in `config/jctagging.yaml` `data.data_dir`.
- TopTagXL: Download 160GB dataset from https://zenodo.org/records/10878355; update the path in `config/toptagxl.yaml` `data.data_dir`.
- Atlas top-tagging: Download 450GB dataset from https://opendata.cern.ch/record/80030; convert it to `root` files using `data/convert_data.py`; then update paths in `config/atlastop.yaml` `data.data_dir`; alternatively download from ITP cluster link (TODO)
- JetSet: Download 230GB dataset from https://opendata.cern.ch/record/93940; convert it to `root` files using `data/convert_data.py`; then update paths in `config/jetset.yaml` `data.data_dir`; alternatively download from ITP cluster link (TODO)

### 3) Train baseline networks

Our baseline networks are defined in `config/model/`. To train them on the different datasets, use the commands below. These are single-epoch trainings on JetClass which are our baseline, well-understood results for multi-epoch trainings and other datasets are on their way.

```bash
python run.py -cp config -cn jetclass model=tr training=jc_5epoch model.net.size=0
python run.py -cp config -cn jetclass model=part training=jc_5epoch model.net.size=0
python run.py -cp config -cn jetclass model=lloca training=jc_5epoch model.net.size=0
python run.py -cp config -cn jetclass model=slim training=jc_5epoch model.net.size=0
python run.py -cp config -cn jetclass model=lgatr training=jc_5epoch model.net.size=0

# repeat for other datasets (repeat with models as above)
python run.py -cp config -cn toptagxl model=tr training=jc_5epoch model.net.size=0
python run.py -cp config -cn atlastop model=tr training=jc_5epoch model.net.size=0
python run.py -cp config -cn jetset model=tr training=jc_5epoch model.net.size=0
```

Comments:

- The above commands load the optimized default parameters. See the files in `config/` for more options.
- By default, all networks except `part` represent jets as sparse objects to avoid zero-padding. This reduces memory usage (2-3x) and yields significant speedups (1.5-2x) for for large networks, but for small networks dense representations can be faster. The key `model.zeropad` controls this.
- `model.net.size` supports continuous values. However, this currently requires `model.zeropad=true` for `tr`, `lloca`, `slim`, and `lgatr` because sparse attention kernels have constraints on the embedding shape. We set `model.net.size=0` here for simplicity.
- Use `data.train_files_range` and `data.fraction_of_file` to control the amount of training data.
- The code supports tracking with `mlflow`, which requires `pip install mlflow` (not just `mlflow-skinny` which is a placeholder) and setting `use_mlflow=true`.
