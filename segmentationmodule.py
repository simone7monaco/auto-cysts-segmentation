from pathlib import Path
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import object_from_dict, find_average, binary_mean_iou

from PIL import Image
import pytorch_lightning as pl
import torchmetrics as tm
import wandb

import numpy as np
import pandas as pd
from time import time


class SegmentCyst(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = self.hparams.model.get('name', '').lower()
        self.model = object_from_dict(hparams["model"])
        
        self.train_images = Path(self.hparams.checkpoint_callback["dirpath"]) / "images/train_predictions"
        self.val_images =  Path(self.hparams.checkpoint_callback["dirpath"]) / "images/val_predictions"

        if not self.hparams.discard_res:
            self.train_images.mkdir(exist_ok=True, parents=True)
            self.val_images.mkdir(exist_ok=True, parents=True)
         
        self.loss = object_from_dict(hparams["loss"])
        self.max_val_iou = 0
        self.timing_result = pd.DataFrame(columns=['name', 'time'])
        self.train_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
            # 'dice': tm.F1Score(task='binary'),
            'pdice': tm.F1Score(task='binary', average='samples'),
        })
        self.val_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
            # 'dice': tm.F1Score(task='binary'),
            'pdice': tm.F1Score(task='binary', average='samples'),
        })
        self.test_metrics = torch.nn.ModuleDict({
            'iou': tm.JaccardIndex(task='binary'),
        })

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch: torch.Tensor, masks: torch.Tensor=None) -> torch.Tensor:
        if masks is not None:
            return self.model(batch, masks)
        else:
            return self.model(batch)

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams.optimizer,
            params=[x for x in self.model.parameters() if x.requires_grad],
        )
        self.optimizers = [optimizer]
        
        if self.hparams.scheduler is not None:
            scheduler = object_from_dict(self.hparams.scheduler, optimizer=optimizer)

            if type(scheduler) == ReduceLROnPlateau:
                    return {
                       'optimizer': optimizer,
                       'lr_scheduler': scheduler,
                       'monitor': 'val_iou'
                   }
            return self.optimizers, [scheduler]
        return self.optimizers
    

    def log_images(self, features, masks, logits_, batch_idx, class_labels={0: "background", 1: "cyst"}):
        for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
            if isinstance(self.trainer.logger, pl.loggers.tensorboard.TensorBoardLogger):
                self.trainer.logger.experiment.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
                self.trainer.logger.experiment.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
                self.trainer.logger.experiment.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0)
            elif isinstance(self.trainer.logger, pl.loggers.wandb.WandbLogger):
                img = wandb.Image(
                    image,
                    masks={
                        "predictions": {
                            "mask_data": y_pred,
                            "class_labels": class_labels,
                        },
                        "groud_truth": {
                            "mask_data": y_true,
                            "class_labels": class_labels,
                        },
                    },
                )
                self.logger.experiment.log({"generated_images": [img]}, commit=False)
            else:
                Image.fromarray(y_pred*255).save(self.train_images/f"{batch_idx}_{img_idx}.png")
                Image.fromarray(y_true*255).save(self.train_images/f"{batch_idx}_{img_idx}_gt.png")
                Image.fromarray(image).save(self.train_images/f"{batch_idx}_{img_idx}_img.png")

    
    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]
                
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            loss = logits['loss']
            logits = logits['pred']
        else:
            logits = self.forward(features)
            loss = self.loss(logits, masks)
        
        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")
        
        if batch_idx == 0 and self.trainer.current_epoch % 2 == 0:
            self.log_images(features, masks, logits_, batch_idx)

        for metric_name, metric in self.train_metrics.items():
            metric(logits, masks.int())
            self.log(f"train_{metric_name}", metric, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss)
        self.log("lr", self._get_current_lr())
        return {"loss": loss}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        
        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
        # result = {}
        
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            loss = logits['loss'] 
            logits = logits['pred']
        else:
            logits = self.forward(features)
            loss = self.loss(logits, masks)
            # result["val_loss"] = loss
            
        logits_ = (logits > 0.5).cpu().detach().numpy().astype("float")

        # result["val_iou"] = binary_mean_iou(logits, masks)
        
        if not self.hparams.discard_res and wandb.run is not None:
            if self.trainer.current_epoch % 5 == 0:
                class_labels = {0: "background", 1: "cyst"}
                mask_img = wandb.Image(
                    features[0, :, :, :],
                    masks={
                        "predictions": {
                            "mask_data": logits_[0, 0, :, :],
                            "class_labels": class_labels,
                        },
                        "groud_truth": {
                            "mask_data": masks.cpu().detach().numpy()[0, 0, :, :],
                            "class_labels": class_labels,
                        },
                    },
                )
                self.logger.experiment.log({"val_images": [mask_img]}, commit=False)

        self.log("val_loss", loss)
        # self.log("val_iou", result["val_iou"])
        for metric_name, metric in self.val_metrics.items():
            metric(logits, masks.int())
            self.log(f"val_{metric_name}", metric, on_step=True, on_epoch=True)

        # self.validation_step_outputs.append(result)

    def on_train_epoch_end(self):
        self.log("epoch", float(self.trainer.current_epoch))

    def test_step(self, batch, batch_id):
        features = batch["features"]
        masks = batch["masks"]
        # result = {}
    
        t0 = time()
        if self.model_name in ['uacanet', 'pranet']:
            logits = self.forward(features, masks)
            logits = logits['pred']
        else:
            logits = self.forward(features)
        
        timing = [time()-t0, features.shape[0]]
        # result["test_iou"] = binary_mean_iou(logits, masks)
        for i in range(features.shape[0]):
            name = batch["image_id"][i]
            logits_ = logits[i][0]

            logits_ = (logits_.cpu().numpy() > self.hparams.test_parameters['threshold']).astype(np.uint8)
            Image.fromarray(logits_*255).save(self.hparams.checkpoint_callback['dirpath'] /'result'/'test'/f"{name}.png")

        self.timing_result.loc[len(self.timing_result)] = timing
        for metric_name, metric in self.test_metrics.items():
            metric(logits, masks.int())
            self.log(f"test_{metric_name}", metric, on_step=True, on_epoch=True)
        # self.test_step_outputs.append(result)
        # return result
    