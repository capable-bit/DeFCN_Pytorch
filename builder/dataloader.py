from torch.utils.data import Dataset
from data.distributed_sampler import DistributedGroupSampler
from data.dataset import COCODataset,ResizeTransform,FlipTransform
import os
import numpy as np
import torch
from pathlib import Path
def build_dataset(cfg,val_train,transforms):
    root = Path(cfg.DATA.COCO_PATH)
    assert root.exists(), f'provided COCO path {root} does not exist'

    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    img_folder, ann_file = PATHS[val_train]

    return COCODataset(ann_file,img_folder,transforms)

def trivial_batch_collator(batch):
    return batch
def worker_init_reset_seed(worker_id):
    from datetime import datetime
    import random

    seed = np.random.randint(2**31) + worker_id
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )

    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)

def build_train_loader(cfg):
    batch_size = cfg.DATA.BATCH_SIZE
    transforms = [ResizeTransform, FlipTransform]

    dataset = build_dataset(cfg,cfg.VAL_TRAIN,transforms)
    print("train_image_num:",len(dataset.dataset_dicts))

    train_sampler = DistributedGroupSampler(dataset, batch_size, cfg.NUM_DEVICES, cfg.RANK)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader

