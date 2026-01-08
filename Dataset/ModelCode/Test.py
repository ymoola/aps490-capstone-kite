import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L  
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning
import pytorchvideo
import os

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
)

def test_dataloader(self):
     
     
        test_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                  ]
                ),
              ),
            ]
        )
        test_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "test.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=test_transform
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

def test_step(self, batch, batch_idx):
    y_hat = self.model(batch["video"])
    loss = F.cross_entropy(y_hat, batch["label"])
    acc = (y_hat.argmax(1) == batch["label"]).float().mean()
    self.log('test_loss', loss)
    self.log('test_acc', acc)



def trainAndTest():
    classification_module = VideoClassificationLightningModule()
    data_module = KineticsDataModule()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, data_module)
    trainer.test(classification_module, datamodule=data_module)