import argparse
import glob
import json
import os
import re


SELECTION_METADATA = {
    'val_loss': 'best.json',
    'classification': 'best_classification.json',
    'retrieval': 'best_retrieval.json',
    'progression': 'best_progression.json',
    'kendall': 'best_kendall.json',
}


def _completed_epoch(path):
    match = re.search(r'epoch=(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else -1


def resolve_checkpoint(run_dir, selection='val_loss'):
    """Resolve one checkpoint from a standardized BYOV training run."""
    run_dir = os.path.abspath(run_dir)
    if selection == 'last':
        candidates = glob.glob(os.path.join(run_dir, 'checkpoints', 'last-epoch=*.ckpt'))
        legacy_last = os.path.join(run_dir, 'checkpoints', 'last.ckpt')
        if os.path.isfile(legacy_last):
            candidates.append(legacy_last)
        if not candidates:
            raise FileNotFoundError(f'No latest checkpoint found in training run: {run_dir}')
        checkpoint = max(candidates, key=_completed_epoch)
        return checkpoint, {
            'selection': selection,
            'best_epoch': _completed_epoch(checkpoint),
            'best_score': None,
            'metadata_file': None,
        }

    if selection not in SELECTION_METADATA:
        raise ValueError(
            f'Unknown checkpoint selection {selection!r}; choose from '
            f'{sorted(SELECTION_METADATA)} or last'
        )
    metadata_path = os.path.join(run_dir, 'metrics', SELECTION_METADATA[selection])
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f'Checkpoint-selection metadata not found: {metadata_path}. '
            'The selected downstream metric may not have been evaluated yet.'
        )
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    recorded_path = metadata.get('checkpoint')
    if not recorded_path:
        raise ValueError(f'No checkpoint path recorded in {metadata_path}')
    checkpoint = (
        recorded_path if os.path.isabs(recorded_path)
        else os.path.join(run_dir, recorded_path)
    )
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f'Checkpoint recorded by {metadata_path} does not exist: {checkpoint}'
        )
    return checkpoint, {
        'selection': selection,
        'best_epoch': metadata.get('best_epoch'),
        'best_score': metadata.get('best_score'),
        'monitor': metadata.get('monitor'),
        'mode': metadata.get('mode'),
        'metadata_file': metadata_path,
    }


def find_latest_training_run(training_root, dataset, selection='val_loss', run_name_filter=''):
    """Find the newest dataset run for which the requested checkpoint exists."""
    dataset_root = os.path.join(os.path.abspath(training_root), dataset)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f'Training dataset directory not found: {dataset_root}')
    candidates = [
        path for path in glob.glob(os.path.join(dataset_root, '*'))
        if os.path.isdir(path)
        and (not run_name_filter or run_name_filter in os.path.basename(path))
    ]
    candidates.sort(key=lambda path: (os.path.getmtime(path), path), reverse=True)
    failures = []
    for run_dir in candidates:
        try:
            resolve_checkpoint(run_dir, selection)
            return os.path.abspath(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))
    detail = f' Last candidate error: {failures[0]}' if failures else ''
    raise FileNotFoundError(
        f'No training run under {dataset_root} provides selection={selection!r}'
        f'{" and filter=" + repr(run_name_filter) if run_name_filter else ""}.{detail}'
    )


def main():
    parser = argparse.ArgumentParser(description='Resolve a checkpoint from BYOV training outputs')
    parser.add_argument('--training-root', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--selection', default='val_loss',
                        choices=[*SELECTION_METADATA, 'last'])
    parser.add_argument('--run-name-filter', default='')
    args = parser.parse_args()
    run_dir = find_latest_training_run(
        args.training_root, args.dataset, args.selection, args.run_name_filter
    )
    print(run_dir)


if __name__ == '__main__':
    main()
