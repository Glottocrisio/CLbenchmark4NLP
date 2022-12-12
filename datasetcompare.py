from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from avalanche.benchmarks.datasets import (
    SimpleDownloadableDataset,
    default_dataset_location,
)
from avalanche.benchmarks.datasets.penn_fudan.penn_fudan_data import (
    penn_fudan_data,
)


def default_mask_loader(mask_path):
    return Image.open(mask_path)


class PennFudanDataset(SimpleDownloadableDataset):
    """
    The Penn-Fudan Pedestrian detection and segmentation dataset
    Adapted from the "TorchVision Object Detection Finetuning Tutorial":
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    """

    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        transform=None,
        loader=default_loader,
        mask_loader=default_mask_loader,
        download=True
    ):
        """
        Creates an instance of the Penn-Fudan dataset.
        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            "pennfudanped" will be used.
        :param transform: The transformation to apply to (img, annotations)
            values.
        :param loader: The image loader to use.
        :param mask_loader: The mask image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location("pennfudanped")

        self.imgs = None
        self.masks = None
        self.targets = None
        self.transform = transform
        self.loader = loader
        self.mask_loader = mask_loader

        super().__init__(
            root,
            penn_fudan_data[0],
            penn_fudan_data[1],
            download=download,
            verbose=True,
        )

        self._load_dataset()

    def _load_metadata(self):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = (self.root / "PennFudanPed" / "PNGImages").iterdir()
        self.masks = (self.root / "PennFudanPed" / "PedMasks").iterdir()

        self.imgs = list(sorted(self.imgs))
        self.masks = list(sorted(self.masks))

        self.targets = [self.make_targets(i) for i in range(len(self.imgs))]
        return Path(self.imgs[0]).exists() and Path(self.masks[0]).exists()

    def make_targets(self, idx):
        # load images and masks
        mask_path = self.masks[idx]

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = self.mask_loader(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return target

    def __getitem__(self, idx):
        target = self.targets[idx]
        img_path = self.imgs[idx]
        img = self.loader(img_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
