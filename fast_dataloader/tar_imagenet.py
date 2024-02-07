import io
import pathlib
import tarfile
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


class TarImageFolder(Dataset):
    def __init__(self,
                 tar_file: pathlib.Path,
                 transform: Callable):
        # tar file is expected to be organized as
        # class-0 / img0
        #         / img1
        #         / ...
        # class-1 / img0
        #         / img1
        #         / ...

        self.tar_file = tar_file
        with tarfile.open(self.tar_file) as tar:
            paths = [pathlib.Path(member.name) for member in tar.getmembers() if member.isfile()]

        self.files = [member for member in paths if (member[1].suffix.lower() in extensions)]
        classes = list({member.parts[0] for member in self.files})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self,
                    item: int
                    ) -> tuple[Any, int]:
        member, path = self.files[item]
        with tarfile.open(self.tar_file) as tar:
            img_file = tar.extractfile(str(path))
            img_data = img_file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        cls = self.class_to_idx[path.parts[0]]
        if self.transform:
            img = self.transform(img)

        return img, cls
