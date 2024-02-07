import pathlib

import torch
import tqdm

from fast_dataloader.cache import Collate, memmap_imagefolder


def main(data_root: str,
         cache_dir: str,
         num_workers: int):
    cache_dir = pathlib.Path(cache_dir)

    data_root = pathlib.Path(data_root)
    train_set, val_set = memmap_imagefolder(data_root / "train", data_root / "val", 384, 256, num_workers, cache_dir,
                                            256)

    collate_fn_train = Collate(device="cuda")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, collate_fn=collate_fn_train)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, collate_fn=collate_fn_train)

    for data in tqdm.tqdm(train_loader):
        pass

    for data in tqdm.tqdm(val_loader):
        pass


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str)
    p.add_argument("--cache_dir", type=str)
    p.add_argument("--num_workers", type=int, default=1)

    args = p.parse_args()

    main(args.data_root, args.cache_dir, args.num_workers)
