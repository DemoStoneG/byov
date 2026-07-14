import json
import logging
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import argparser
from utils.experiment import prepare_experiment, save_run_metadata, setup_logging


logger = logging.getLogger(__name__)


def prepare_data_loader(args, mode, batch_size=1, num_workers=0):
    from torch.utils.data import DataLoader
    from dataset.video_align_dataset import VideoAlignmentDownstreamDataset

    dataset = VideoAlignmentDownstreamDataset(args, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    logger.info('Data loader %s len %s', mode, len(data_loader))
    return data_loader, dataset


def extract_embedding(mode, data_loader, base_model, encoder, save_path, device):
    import numpy as np
    import torch
    from tqdm import tqdm

    os.makedirs(save_path, exist_ok=True)
    embeds_list = []
    labels_list = []
    for batch in tqdm(data_loader, desc=f'Extract {mode}'):
        frame, frame_label, _video_path = batch
        frame = frame.reshape(1, -1, *frame.shape[-3:])
        frame = frame.permute(0, 1, 4, 2, 3).float().to(device)
        with torch.no_grad():
            embeds = base_model(frame)
            embeds, _, _, _, _, _, _ = encoder(embeds)
        # Only remove the batch dimension. Generic squeeze() breaks one-frame videos.
        embeds_list.append(embeds.squeeze(0).cpu().numpy())
        labels_list.append(np.asarray([label.numpy() for label in frame_label]).reshape(-1))

    if not embeds_list:
        raise RuntimeError(f'No videos were available while extracting the {mode} split')

    embeds = np.concatenate(embeds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    if embeds.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f'Embedding/label count mismatch for {mode}: {embeds.shape[0]} vs {labels.shape[0]}'
        )

    embeds_path = os.path.join(save_path, f'{mode}_embeds.npy')
    labels_path = os.path.join(save_path, f'{mode}_label.npy')
    np.save(embeds_path, embeds)
    np.save(labels_path, labels)
    logger.info('Saved %s embeddings %s to %s', mode, tuple(embeds.shape), embeds_path)
    logger.info('Saved %s labels %s to %s', mode, tuple(labels.shape), labels_path)
    return {
        'embeddings': embeds_path,
        'labels': labels_path,
        'embedding_shape': list(embeds.shape),
        'label_shape': list(labels.shape),
    }


def _resolve_device(requested):
    import torch

    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device {requested} was requested, but CUDA is unavailable')
    return device


def _validate_backbone_config(base_model, args):
    config = base_model.base_model.config
    hidden_size = int(config.hidden_size)
    image_size = int(config.image_size)
    patch_size = int(config.patch_size)
    num_tokens = (image_size // patch_size) ** 2
    errors = []
    if hidden_size != args.hidden_dim:
        errors.append(f'hidden_dim={args.hidden_dim}, backbone hidden_size={hidden_size}')
    if num_tokens != args.num_tokens:
        errors.append(f'num_tokens={args.num_tokens}, backbone patch tokens={num_tokens}')
    if args.input_size != image_size:
        errors.append(f'input_size={args.input_size}, backbone image_size={image_size}')
    if errors:
        raise ValueError('Backbone/probe configuration mismatch: ' + '; '.join(errors))
    return {
        'model_type': config.model_type,
        'image_size': image_size,
        'patch_size': patch_size,
        'num_tokens': num_tokens,
        'hidden_size': hidden_size,
        'probe_embedding_size': args.embedding_size,
    }


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info('Saved %s', path)


def _update_comparison_summary(output_root, dataset, metrics):
    summary_dir = os.path.join(output_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    dataset_path = os.path.join(summary_dir, f'{dataset}_{metrics["split"]}.json')
    _write_json(dataset_path, metrics)

    combined_path = os.path.join(summary_dir, 'all_results.json')
    combined = {}
    if os.path.isfile(combined_path):
        try:
            with open(combined_path, 'r') as f:
                combined = json.load(f)
        except (json.JSONDecodeError, OSError):
            combined = {}
    combined[dataset] = metrics
    _write_json(combined_path, combined)
    return dataset_path, combined_path


def _validate_eval_plan(args):
    if args.eval_mode not in ('val', 'test'):
        raise ValueError('--eval_mode must be val or test')
    unknown_tasks = set(args.eval_task) - set('1234')
    if unknown_tasks:
        raise ValueError(f'Unknown eval task identifiers: {sorted(unknown_tasks)}')
    fitted_tasks = sorted(set(args.eval_task) & set('13'))
    if args.no_downstream_fit and fitted_tasks:
        raise ValueError(
            '--no_downstream_fit forbids task 1 (SVM classification) and task 3 '
            f'(linear phase regression); requested fitted tasks: {fitted_tasks}. Use --eval_task 24.'
        )
    if not args.eval_task:
        raise ValueError('--eval_task cannot be empty')
    if args.embedding_file_split and not args.embedding_dir:
        raise ValueError('--embedding_file_split is only valid with --embedding_dir')


def _validate_precomputed_files(embeddings_dir, file_split, dataset_eval, dataset_train=None):
    import numpy as np

    result = {}
    observed_embedding_dim = None
    splits = [(file_split, dataset_eval)]
    if dataset_train is not None:
        splits.append(('train', dataset_train))
    for filename_split, dataset in splits:
        embeds_path = os.path.join(embeddings_dir, f'{filename_split}_embeds.npy')
        labels_path = os.path.join(embeddings_dir, f'{filename_split}_label.npy')
        embeds = np.load(embeds_path, mmap_mode='r')
        labels = np.load(labels_path, mmap_mode='r')
        expected_frames = int(sum(dataset.video_len_list))
        if embeds.ndim != 2:
            raise ValueError(f'{embeds_path} must be 2-D, got shape {embeds.shape}')
        if observed_embedding_dim is None:
            observed_embedding_dim = int(embeds.shape[1])
        elif embeds.shape[1] != observed_embedding_dim:
            raise ValueError(
                f'Precomputed embedding dimensions differ across splits: {embeds_path} has '
                f'{embeds.shape[1]} dimensions, while another split has '
                f'{observed_embedding_dim}.'
            )
        if embeds.shape[0] != expected_frames or labels.reshape(-1).shape[0] != expected_frames:
            raise ValueError(
                f'Precomputed {filename_split} files do not match the selected dataset split: '
                f'expected {expected_frames} frames, embeddings={embeds.shape[0]}, '
                f'labels={labels.reshape(-1).shape[0]}. Check --eval_mode and '
                '--embedding_file_split.'
            )
        result[filename_split] = {
            'embedding_shape': list(embeds.shape),
            'label_shape': list(labels.shape),
            'expected_frames': expected_frames,
        }
    result['observed_embedding_dim'] = observed_embedding_dim
    return result


def main():
    args = argparser.parse_args()
    _validate_eval_plan(args)
    args, _is_resume = prepare_experiment(args)
    setup_logging(args.logs_dir, filename='eval.log')
    save_run_metadata(args, sys.argv, is_resume=False)

    embeddings_dir = (
        os.path.abspath(args.embedding_dir)
        if args.embedding_dir
        else os.path.join(args.artifacts_dir, 'embeddings')
    )
    embedding_file_split = args.embedding_file_split or args.eval_mode
    plan = {
        'mode': args.eval_mode,
        'embedding_file_split': embedding_file_split,
        'tasks': list(args.eval_task),
        'no_downstream_fit': bool(args.no_downstream_fit),
        'checkpoint': os.path.abspath(args.ckpt) if args.ckpt else '',
        'vision_encoder_path': os.path.abspath(args.vision_encoder_path),
        'embeddings_dir': embeddings_dir,
        'uses_precomputed_embeddings': bool(args.embedding_dir and not args.extract_embedding),
        'metrics_file': os.path.join(args.metrics_dir, f'{args.eval_mode}.json'),
    }
    _write_json(os.path.join(args.config_dir, 'evaluation_plan.json'), plan)
    logger.info('Evaluation run directory: %s', args.run_dir)

    if args.dry_run == 'config':
        logger.info('Evaluation config dry run complete; no model, checkpoint, or dataset was loaded')
        return

    if args.ckpt and not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')

    from evaluation.classification import classification
    from evaluation.frame_retrieval import frame_retrieval
    from evaluation.event_completion import compute_progression_value
    from evaluation.kendalls_tau import kendalls_tau

    device = None
    needs_train_split = bool(set(args.eval_task) & set('13'))
    loader_train = dataset_train = None
    if needs_train_split:
        loader_train, dataset_train = prepare_data_loader(
            args, 'train', batch_size=1, num_workers=args.num_workers
        )
    loader_eval, dataset_eval = prepare_data_loader(
        args, args.eval_mode, batch_size=1, num_workers=args.num_workers
    )

    extraction = {}
    backbone_info = None
    load_report = None
    if args.extract_embedding:
        if not args.ckpt:
            raise ValueError('--ckpt is required when --extract_embedding is used')
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')
        if args.embedding_dir:
            raise ValueError('Do not combine --extract_embedding with --embedding_dir')

        from models.embedder import Embedder, byov_encoder
        from utils.load_model import load_ckpt

        device = _resolve_device(args.device)
        logger.info('Using device: %s', device)
        encoder = byov_encoder(args).to(device)
        encoder.eval()
        load_report = load_ckpt(encoder, args.ckpt)
        _write_json(os.path.join(args.config_dir, 'checkpoint_load.json'), load_report)
        base_model = Embedder(args).to(device)
        base_model.eval()
        backbone_info = _validate_backbone_config(base_model, args)

        if needs_train_split:
            extraction['train'] = extract_embedding(
                'train', loader_train, base_model, encoder, embeddings_dir, device
            )
        extraction[args.eval_mode] = extract_embedding(
            args.eval_mode, loader_eval, base_model, encoder, embeddings_dir, device
        )
    else:
        required = [
            os.path.join(embeddings_dir, f'{embedding_file_split}_embeds.npy'),
            os.path.join(embeddings_dir, f'{embedding_file_split}_label.npy'),
        ]
        if needs_train_split:
            required.extend([
                os.path.join(embeddings_dir, 'train_embeds.npy'),
                os.path.join(embeddings_dir, 'train_label.npy'),
            ])
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                'Pre-extracted embeddings are missing; pass --extract_embedding. Missing: '
                + ', '.join(missing)
            )
        extraction['precomputed_validation'] = _validate_precomputed_files(
            embeddings_dir,
            embedding_file_split,
            dataset_eval,
            dataset_train if needs_train_split else None,
        )

    observed_embedding_dim = (
        extraction.get('precomputed_validation', {}).get('observed_embedding_dim')
        if not args.extract_embedding
        else args.embedding_size
    )
    metrics = {
        'dataset': args.dataset,
        'split': args.eval_mode,
        'checkpoint': os.path.abspath(args.ckpt) if args.ckpt else None,
        'embedding_source': embeddings_dir,
        'parameters': {
            'tasks': list(args.eval_task),
            'embedding_file_split': embedding_file_split,
            'uses_precomputed_embeddings': bool(args.embedding_dir and not args.extract_embedding),
            'fits_downstream_svm': '1' in args.eval_task,
            'fits_downstream_linear_regressor': '3' in args.eval_task,
            'device': str(device) if device is not None else None,
            'base_model_name': args.base_model_name,
            'backbone_frozen': bool(args.freeze_base),
            'vision_encoder_path': os.path.abspath(args.vision_encoder_path),
            'input_size': int(args.input_size),
            'num_frames': int(args.num_frames),
            'num_tokens': int(args.num_tokens),
            'backbone_hidden_dim': int(args.hidden_dim),
            'configured_probe_embedding_size': int(args.embedding_size),
            'evaluated_embedding_size': int(observed_embedding_dim),
            'token_selection_ratio': float(args.topk_ratio),
            'msm_mask_ratio': float(args.mask_ratio),
            'mcm_mask_ratio': float(args.mask_ratio * 2),
        },
        'data': extraction,
    }
    if backbone_info is not None:
        metrics['backbone'] = backbone_info
    if load_report is not None:
        metrics['checkpoint_load'] = load_report

    if '1' in args.eval_task:
        regular, ego2exo, exo2ego = classification(
            embeddings_dir,
            dataset_train.video_ego_id,
            dataset_eval.video_ego_id,
            eval_mode=embedding_file_split,
        )
        metrics['classification'] = {
            'regular_f1': float(regular),
            'ego2exo_f1': float(ego2exo),
            'exo2ego_f1': float(exo2ego),
            'fits_svm': True,
        }

    if '2' in args.eval_task:
        regular, ego2exo, exo2ego = frame_retrieval(
            embeddings_dir,
            dataset_eval.video_len_list,
            dataset_eval.video_paths1,
            eval_mode=embedding_file_split,
        )
        metrics['retrieval'] = {
            'regular_map10': float(regular),
            'ego2exo_map10': float(ego2exo),
            'exo2ego_map10': float(exo2ego),
        }

    if '3' in args.eval_task:
        train_score, eval_score = compute_progression_value(
            embeddings_dir,
            dataset_train.video_len_list,
            dataset_eval.video_len_list,
            modify_embeddings=args.dataset == 'pour_liquid',
            eval_mode=embedding_file_split,
        )
        metrics['progression'] = {
            'train_r2': float(train_score),
            'eval_r2': float(eval_score),
            'fits_linear_regressor': True,
        }

    if '4' in args.eval_task:
        eval_tau = kendalls_tau(
            embeddings_dir,
            dataset_eval.video_len_list,
            dataset_eval.video_paths1,
            embedding_file_split,
            False,
        )
        metrics['kendall'] = {'eval_tau': float(eval_tau)}

    metrics_path = os.path.join(args.metrics_dir, f'{args.eval_mode}.json')
    _write_json(metrics_path, metrics)
    dataset_summary_path, combined_summary_path = _update_comparison_summary(
        args.output_root, args.dataset, metrics
    )
    logger.info('Evaluation complete: %s', json.dumps(metrics, ensure_ascii=False))
    logger.info('Dataset summary: %s', dataset_summary_path)
    logger.info('Combined summary: %s', combined_summary_path)


if __name__ == '__main__':
    main()
