from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from torch.utils.data import Dataset, DataLoader

import albumentations as albu
from albumentations.core.serialization import from_dict

from skimage.io import imread
from utils import get_samples, split_dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        transform: albu.Compose,
        noG=False,
    ):
        self.samples = samples
        self.transform = transform
        self.length = len(self.samples)
        self.noG = noG

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        image_path, mask_path = self.samples[idx]

        image = imread(image_path)
        mask = imread(mask_path)

        if self.noG: # TODO: What if done after the augmentation the image?
            image[:, :, 1] = 0

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id": image_path.stem,
            "features": image_to_tensor(image),
            "masks": torch.unsqueeze(mask, 0).float(),
        }

class CystDataModule(pl.LightningDataModule):
    def __init__(self, verbose=True, **hparams):
        super().__init__()
        
        for k in ['model', 'optimizer', 'loss']:
            hparams.pop(k)
        self.save_hyperparameters()
        self.verbose = verbose

        if not self.hparams['image_path'].exists():
            raise ValueError(f"Image path {self.hparams['image_path']} does not exist. Perform a run using Wandb to get the dataset or download it from another source.")

        splits = split_dataset(self.hparams)
        self.train_samples=splits['train']
        self.val_samples=splits['valid']
        self.test_samples=splits['test']

        if self.train_samples is None:
            samples = get_samples(hparams["image_path"], hparams["mask_path"])
            num_train = int((1 - hparams["val_split"]) * len(samples))
            self.train_samples = samples[:num_train]
            self.val_samples = samples[num_train:]
        
        if self.verbose:
            print("Len train samples = ", len(self.train_samples))
            print("Len val samples = ", len(self.val_samples))

        self.train_aug = from_dict(self.hparams.train_aug)
        self.val_aug = from_dict(self.hparams.val_aug)
        self.test_aug = from_dict(self.hparams.test_aug)

        self.batch_size = self.hparams.train_parameters["batch_size"]
        self.val_batch_size = self.hparams.val_parameters["batch_size"]

    def train_dataloader(self):
        result = DataLoader(
            SegmentationDataset(self.train_samples, self.train_aug, noG=self.hparams["noG_preprocessing"]),
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )
        if self.verbose:
            print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        result = DataLoader(
            SegmentationDataset(self.val_samples, self.val_aug, noG=self.hparams["noG_preprocessing"]),
            batch_size=self.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
        )
        if self.verbose:
            print("Val dataloader = ", len(result))
        return result

    def test_dataloader(self):
        result = DataLoader(
            SegmentationDataset(self.test_samples, self.test_aug, noG=self.hparams["noG_preprocessing"]),
            batch_size=self.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
        ) if self.test_samples is not None else []
        if self.verbose:
            print("Test dataloader = ", len(result))
        return result
