# inference/main.py
from __future__ import annotations

import os

from ctr_gcn import TrainConfig, train_validate_test


# ============================================================
# EDIT THESE VARIABLES
# ============================================================

# ---- Dataset location (your new structure)
DATASET_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\dataset_ctr_gcn"   # contains train.npz / val.npz / test.npz
TRAIN_NPZ = os.path.join(DATASET_DIR, "train.npz")
VAL_NPZ   = os.path.join(DATASET_DIR, "val.npz")
TEST_NPZ  = os.path.join(DATASET_DIR, "test.npz")

# ---- CTR-GCN repo location
CTR_GCN_REPO = r"CV\frameworks\CTR-GCN"

# ---- Training output
RUN_OUT_DIR = r"CV\runs\ctr_gcn_run1"

# ---- Model/data shape
NUM_CLASS = 2        # pass/fail as 2-class CE
NUM_POINT = 17       # COCO joints
NUM_PERSON = 1
IN_CHANNELS = 3      # x,y,conf

# NOTE:
# CTR-GCN repos often default graph to NTU (25 joints).
# We'll likely need a COCO graph definition inside the CTR-GCN repo.
# For now we leave it as a variable so it's easy to patch once we confirm the repo.
GRAPH = "graph.coco17.Graph"
GRAPH_ARGS = {}      # we will set COCO layout here later once we implement/confirm it

DROPOUT = 0.0

# ---- Class imbalance emphasis
USE_WEIGHTED_SAMPLER = True        # balances batches by oversampling minority class
USE_CLASS_WEIGHTED_LOSS = True     # weights CE loss inversely to class frequency

# ---- Train hyperparams TODO: optimize
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

DEVICE = "cuda"  # or "cpu" or "cuda:0"

# ============================================================


def main():
    # Sanity: dataset exists
    for p in (TRAIN_NPZ, VAL_NPZ, TEST_NPZ):
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing dataset file: {os.path.abspath(p)}")

    cfg = TrainConfig(
        device=DEVICE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER,
        use_class_weighted_loss=USE_CLASS_WEIGHTED_LOSS,
        out_dir=RUN_OUT_DIR,
        save_best=True,
        best_metric="val_balanced_acc",
    )

    model_kwargs = dict(
        num_class=NUM_CLASS,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        in_channels=IN_CHANNELS,
        graph=GRAPH,
        graph_args=GRAPH_ARGS,
        drop_out=DROPOUT,
    )

    train_validate_test(
        ctr_repo_root=CTR_GCN_REPO,
        train_npz=TRAIN_NPZ,
        val_npz=VAL_NPZ,
        test_npz=TEST_NPZ,
        model_kwargs=model_kwargs,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
