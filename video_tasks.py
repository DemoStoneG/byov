import copy
import glob
import json
import logging
import math
import os
import re
import shutil
import numpy as np
import torch
import random
from pytorch_lightning.core import LightningModule
from dataset.video_align_dataset import VideoAlignmentTrainDataset
from models.embedder import Embedder, byov_encoder, byov_decoder, validate_backbone_config
from evaluation.evaluate_features import prepare_data_loader, extract_embedding
from evaluation.classification import classification
from evaluation.frame_retrieval import frame_retrieval
from evaluation.event_completion import compute_progression_value
from evaluation.kendalls_tau import kendalls_tau
from utils.pos_embed import interpolate_pos_embed

logger = logging.getLogger(__name__)


DOWNSTREAM_RETENTION_METRICS = {
    'classification': ('classification', 'regular_f1'),
    'retrieval': ('retrieval', 'regular_map10'),
    'progression': ('progression', 'val_score'),
    'kendall': ('kendall', 'val_tau'),
}


def prune_downstream_embeddings(metrics_dir, artifacts_dir):
    """Keep only the embedding epochs that are currently best for a downstream task."""
    best_by_task = {}
    metric_paths = sorted(glob.glob(os.path.join(metrics_dir, 'downstream_epoch_*.json')))
    for metrics_path in metric_paths:
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            epoch = int(metrics['epoch'])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning('Skipping invalid downstream metrics file %s: %s', metrics_path, exc)
            continue
        for task, (section, score_key) in DOWNSTREAM_RETENTION_METRICS.items():
            try:
                score = float(metrics[section][score_key])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(score):
                logger.warning(
                    'Ignoring non-finite %s score in %s: %s', task, metrics_path, score
                )
                continue
            previous = best_by_task.get(task)
            # Strict comparison matches save_top_k=1 behavior and keeps the earlier epoch on ties.
            if previous is None or score > previous['score']:
                best_by_task[task] = {
                    'epoch': epoch,
                    'score': score,
                    'metrics_file': os.path.basename(metrics_path),
                }

    keep_epochs = {record['epoch'] for record in best_by_task.values()}
    embeddings_root = os.path.join(artifacts_dir, 'embeddings')
    removed = []
    if os.path.isdir(embeddings_root):
        for path in glob.glob(os.path.join(embeddings_root, 'epoch_*')):
            if not os.path.isdir(path):
                continue
            match = re.fullmatch(r'epoch_(\d+)', os.path.basename(path))
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch not in keep_epochs:
                shutil.rmtree(path)
                removed.append(os.path.basename(path))

    retention = {
        'policy': 'keep_current_best_epoch_per_downstream_task',
        'best_by_task': best_by_task,
        'kept_embedding_epochs': sorted(keep_epochs),
        'removed_embedding_directories': removed,
    }
    retention_path = os.path.join(metrics_dir, 'embedding_retention.json')
    with open(retention_path, 'w') as f:
        json.dump(retention, f, indent=4)
    logger.info(
        'Embedding retention kept epochs %s and removed %s',
        retention['kept_embedding_epochs'],
        removed,
    )
    return retention


class VideoAlignment(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Embedder(args)
        backbone_info = validate_backbone_config(self.model, args)
        logger.info('Validated backbone configuration: %s', backbone_info)
        with open(os.path.join(args.config_dir, 'backbone.json'), 'w') as f:
            json.dump(backbone_info, f, indent=2)
        self.encoder = byov_encoder(args)
        self.decoder = byov_decoder(args)
        self.checkpoint_metric = "train_loss"
        self.data_path = None
        self.ds_loader_train = self.ds_dataset_train = None
        self.ds_loader_val = self.ds_dataset_val = None
        if args.ds_every_n_epoch > 0:
            self.ds_loader_train, self.ds_dataset_train = prepare_data_loader(args, 'train', batch_size=1)
            self.ds_loader_val, self.ds_dataset_val = prepare_data_loader(args, 'val', batch_size=1)
            logger.info('Constructed downstream loaders: train=%s val=%s', len(self.ds_loader_train), len(self.ds_loader_val))
        else:
            logger.info('Periodic downstream evaluation disabled; labels will not be loaded')

    def _extract_clip_features(self, frames):
        if self.args.freeze_base:
            with torch.no_grad():
                return self.model(frames)
        return self.model(frames)


    def training_step(self, batch, batch_idx):
        frames, steps, seq_lens = batch
        x1 = frames[:, 0, ...].permute(0, 1, 4, 2, 3)  # (bs, 32, 3, 224, 224)
        x2 = frames[:, 1, ...].permute(0, 1, 4, 2, 3)
        x1 = self._extract_clip_features(x1).detach()    # [bs, ts, hidden_dim]
        x2 = self._extract_clip_features(x2).detach()    # [bs, ts, hidden_dim]
        
        embed_ref = torch.cat([x1, x2], dim=1)

        z1, embeds1_r1, embeds1_r2, mask1_r1, mask1_r2, ids_restore1_r1, ids_restore1_r2 = self.encoder(x1)
        z2, embeds2_r1, embeds2_r2, mask2_r1, mask2_r2, ids_restore2_r1, ids_restore2_r2 = self.encoder(x2)
        # embed_ref = torch.cat([embeds1.clone().detach(), embeds2.clone().detach()], dim=1)
        
        mask = torch.zeros((z1.shape[0], z1.shape[1]), device=z1.device)
        
        embed_pred, embed1_pred, embed2_pred = self.decoder(z1, embeds1_r1, embeds1_r2, z2, embeds2_r1, embeds2_r2,
                                                            ids_restore1_r1, ids_restore1_r2, ids_restore2_r1, ids_restore2_r2)
        
        mask1 = torch.cat([mask1_r1, mask2_r1], dim=1)
        loss1 = (embed_pred - embed_ref) ** 2
        loss1 = loss1.mean(dim=-1)
        loss1 = (loss1 * mask1).sum() / mask1.sum()
        
        mask2 = torch.cat([mask1_r2, mask], dim=1)
        loss2 = (embed1_pred - embed_ref) ** 2
        loss2 = loss2.mean(dim=-1)
        loss2 = (loss2 * mask2).sum() / mask2.sum()
        
        mask3 = torch.cat([mask, mask2_r2], dim=1)
        loss3 = (embed2_pred - embed_ref) ** 2
        loss3 = loss3.mean(dim=-1)
        loss3 = (loss3 * mask3).sum() / mask3.sum()

        # loss = loss1 #loss2 + loss3
        loss = loss1 + loss2 + loss3 
        self.log('train/loss_msm', loss1, on_step=False, on_epoch=True)
        self.log('train/loss_mcm_view1', loss2, on_step=False, on_epoch=True)
        self.log('train/loss_mcm_view2', loss3, on_step=False, on_epoch=True)
        self.log('train/loss_step', loss, on_step=True, on_epoch=False)
        self.log('train/loss_epoch', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frames, steps, seq_lens = batch
        x1 = frames[:, 0, ...].permute(0, 1, 4, 2, 3)  # (bs, 64, 3, 168, 168)
        x2 = frames[:, 1, ...].permute(0, 1, 4, 2, 3)
        x1 = self._extract_clip_features(x1).detach()    # [bs, ts, hidden_dim]
        x2 = self._extract_clip_features(x2).detach()    # [bs, ts, hidden_dim]
        embed_ref = torch.cat([x1, x2], dim=1)
        
        # Forwarding to encoder
        z1, embeds1_r1, embeds1_r2, mask1_r1, mask1_r2, ids_restore1_r1, ids_restore1_r2 = self.encoder(x1)
        z2, embeds2_r1, embeds2_r2, mask2_r1, mask2_r2, ids_restore2_r1, ids_restore2_r2 = self.encoder(x2)
                
        # Forwarding to decoder
        embed_pred, embed1_pred, embed2_pred = self.decoder(z1, embeds1_r1, embeds1_r2, z2, embeds2_r1, embeds2_r2,
                                                            ids_restore1_r1, ids_restore1_r2, ids_restore2_r1, ids_restore2_r2)

        mask = torch.zeros((z1.shape[0], z1.shape[1]), device=z1.device)

        mask1 = torch.cat([mask1_r1, mask2_r1], dim=1)
        loss1 = (embed_ref - embed_pred) ** 2
        loss1 = loss1.mean(dim=-1)
        loss1 = (loss1 * mask1).sum() / mask1.sum()
        
        mask2 = torch.cat([mask1_r2, mask], dim=1)
        loss2 = (embed_ref - embed1_pred) ** 2
        loss2 = loss2.mean(dim=-1)
        loss2 = (loss2 * mask2).sum() / mask2.sum()
        
        mask3 = torch.cat([mask, mask2_r2], dim=1)
        loss3 = (embed_ref - embed2_pred) ** 2
        loss3 = loss3.mean(dim=-1)
        loss3 = (loss3 * mask3).sum() / mask3.sum()

        loss = loss1 + loss2 + loss3 
        embeddings = torch.stack((z1, z2), dim=1)  # (bs, 2, 32, 256)

        self.log('val/loss_msm', loss1, on_step=False, on_epoch=True)
        self.log('val/loss_mcm_view1', loss2, on_step=False, on_epoch=True)
        self.log('val/loss_mcm_view2', loss3, on_step=False, on_epoch=True)
        self.log('completed_epochs', float(self.current_epoch + 1), on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.evaluate_downstream(batch_idx, embeddings.device)
        return loss

    def evaluate_downstream(self, batch_idx, device):
        if self.args.ds_every_n_epoch > 0 and self.global_rank == 0 and batch_idx == 0 and (
                self.current_epoch + 1) % self.args.ds_every_n_epoch == 0:
            completed_epoch = int(self.current_epoch + 1)
            epoch_name = f'epoch_{completed_epoch:03d}'
            embedding_dir = os.path.join(self.args.artifacts_dir, 'embeddings', epoch_name)
            os.makedirs(embedding_dir, exist_ok=True)
            extract_embedding('train', self.ds_loader_train, self.model, self.encoder, embedding_dir, device)
            extract_embedding('val', self.ds_loader_val, self.model, self.encoder, embedding_dir, device)
            metrics = {
                'epoch': completed_epoch,
            }

            if '1' in self.args.eval_task:  # classification
                regular_f1, ego2exo_val_f1, exo2ego_val_f1 = classification(embedding_dir,
                                                                            self.ds_dataset_train.video_ego_id,
                                                                            self.ds_dataset_val.video_ego_id)
                metrics['classification'] = {
                    'regular_f1': float(regular_f1),
                    'ego2exo_val_f1': float(ego2exo_val_f1),
                    'exo2ego_val_f1': float(exo2ego_val_f1),
                }
                self.log('classification/regular_f1', regular_f1, on_step=False, on_epoch=True)
                self.log('classification/ego2exo_val_f1', ego2exo_val_f1, on_step=False, on_epoch=True)
                self.log('classification/exo2ego_val_f1', exo2ego_val_f1, on_step=False, on_epoch=True)
                self.log('checkpoint_classification', regular_f1, on_step=False, on_epoch=True)

            if '2' in self.args.eval_task:  # retrieval
                regular_map10, ego2exo_val_map10, exo2ego_val_map10 = frame_retrieval(embedding_dir,
                                                                                      self.ds_dataset_val.video_len_list,
                                                                                      self.ds_dataset_val.video_paths1)
                metrics['retrieval'] = {
                    'regular_map10': float(regular_map10),
                    'ego2exo_val_map10': float(ego2exo_val_map10),
                    'exo2ego_val_map10': float(exo2ego_val_map10),
                }
                self.log('retrieval/regular_map10', float(regular_map10), on_step=False, on_epoch=True)
                self.log('retrieval/ego2exo_val_map10', float(ego2exo_val_map10), on_step=False, on_epoch=True)
                self.log('retrieval/exo2ego_val_map10', float(exo2ego_val_map10), on_step=False, on_epoch=True)
                self.log('checkpoint_retrieval', float(regular_map10), on_step=False, on_epoch=True)

            if '3' in self.args.eval_task:  # event completion
                modify_embeddings = True if self.args.dataset == 'pour_liquid' else False  # augment embedding for pour_liquid
                train_score, val_score = compute_progression_value(embedding_dir, self.ds_dataset_train.video_len_list,
                                                                   self.ds_dataset_val.video_len_list, modify_embeddings)
                metrics['progression'] = {
                    'train_score': float(train_score),
                    'val_score': float(val_score),
                }
                self.log('progression/train_score', train_score, on_step=False, on_epoch=True)
                self.log('progression/val_score', val_score, on_step=False, on_epoch=True)
                self.log('checkpoint_progression', val_score, on_step=False, on_epoch=True)

            if '4' in self.args.eval_task:  # kendall's tau
                train_tau = kendalls_tau(embedding_dir, self.ds_dataset_train.video_len_list,
                                         self.ds_dataset_train.video_paths1, 'train', False)
                val_tau = kendalls_tau(embedding_dir, self.ds_dataset_val.video_len_list,
                                       self.ds_dataset_val.video_paths1, 'val', False)
                metrics['kendall'] = {
                    'train_tau': float(train_tau),
                    'val_tau': float(val_tau),
                }
                self.log('kendall/train_tau', train_tau, on_step=False, on_epoch=True)
                self.log('kendall/val_tau', val_tau, on_step=False, on_epoch=True)
                self.log('checkpoint_kendall', val_tau, on_step=False, on_epoch=True)

            metrics_path = os.path.join(self.args.metrics_dir, f'downstream_{epoch_name}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info('Saved downstream metrics to %s', metrics_path)
            prune_downstream_embeddings(self.args.metrics_dir, self.args.artifacts_dir)

    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def train_dataloader(self):
        dataset = VideoAlignmentTrainDataset(self.args, 'train')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader

    def val_dataloader(self):
        dataset = VideoAlignmentTrainDataset(self.args, 'val')
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers)
        return loader
