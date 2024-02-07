import pathlib

import torch
import tqdm
from tensordict import MemoryMappedTensor
from tensordict.prototype import tensorclass
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms


@tensorclass
class ImageNetData:
    # based on https://pytorch.org/tensordict/tutorials/tensorclass_imagenet.html
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls,
                     dataset: ImageFolder,
                     num_workers: int,
                     memmap_dir: pathlib.Path,
                     batch_size: int):
        data = cls(
            images=MemoryMappedTensor.empty((len(dataset), *dataset[0][0].squeeze().shape,), dtype=torch.uint8),
            targets=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
            batch_size=[len(dataset)])

        # locks the tensorclass and ensures that is_memmap will return True.
        data.memmap_(memmap_dir, num_threads=num_workers)

        dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        i = 0
        for image, target in tqdm.tqdm(dl):
            _batch = image.shape[0]
            data[i: i + _batch] = cls(images=image, targets=target, batch_size=[_batch])
            i += _batch

        return data


def memmap_imagefolder(train_root: pathlib.Path,
                       val_root: pathlib.Path,
                       train_resize: int,
                       val_resize: int,
                       num_workers: int,
                       memmap_dir: pathlib.Path,
                       batch_size: int,
                       ) -> tuple[ImageNetData, ImageNetData, int]:
    train_resize = (train_resize, train_resize)
    val_resize = (val_resize, val_resize)
    train_set = ImageFolder(train_root,
                            transform=transforms.Compose([transforms.ToImage(),
                                                          transforms.ToDtype(torch.uint8, scale=True),
                                                          transforms.Resize(train_resize)]))

    train_set = ImageNetData.from_dataset(train_set, num_workers=num_workers, memmap_dir=memmap_dir / "train",
                                          batch_size=batch_size)
    val_set = ImageFolder(val_root,
                          transform=transforms.Compose([transforms.ToImage(),
                                                        transforms.ToDtype(torch.uint8, scale=True),
                                                        transforms.Resize(val_resize)]))
    val_set = ImageNetData.from_dataset(val_set, num_workers=num_workers, memmap_dir=memmap_dir / "val",
                                        batch_size=batch_size)

    return train_set, val_set


class Collate(nn.Module):
    def __init__(self,
                 transform=None,
                 device=None):
        super().__init__()
        self.transform = transform
        self.device = torch.device(device)

    def __call__(self,
                 x: ImageNetData
                 ) -> ImageNetData:
        # move data to RAM
        if self.device.type == "cuda":
            out = x.pin_memory()
        else:
            out = x
        if self.device:
            # move data to gpu
            out = out.to(self.device, non_blocking=True)
        if self.transform:
            # apply transforms on gpu
            out.images = self.transform(out.images)
        return out
