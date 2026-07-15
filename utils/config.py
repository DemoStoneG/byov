import argparse

argparser = argparse.ArgumentParser(description='BYOV training and downstream evaluation')
# Training
argparser.add_argument('--num_gpus', type=int, default=1, help='gpus')
argparser.add_argument('--task', type=str, default='align', help='Tasks: align or align_bbox')
argparser.add_argument('--eval_only', action='store_true', help='eval only')
argparser.add_argument('--output_dir', type=str, default='debug', help='Path to results')
argparser.add_argument('--output_root', type=str, default='/mnt/data/wzh/experiments/byov', help='Root directory for experiment outputs')
argparser.add_argument('--run_name', type=str, default='debug', help='Human-readable run name')
argparser.add_argument('--resume', type=str, default='', help='Existing run directory to resume from its newest last-epoch checkpoint')
argparser.add_argument('--seed', type=int, default=42, help='random seed')
argparser.add_argument('--dry_run', type=str, default='', choices=['', 'config'], help='Run lightweight checks without importing training dependencies')
argparser.add_argument('--smoke_test', action='store_true', help='Run one train and validation batch for one epoch')
argparser.add_argument('--ds_every_n_epoch', type=int, default=10, help='downstream evaluation every n epochs')
argparser.add_argument('--save_every', type=int, default=0,
                       help='save periodic epoch checkpoints every n epochs; 0 disables them')
argparser.add_argument('--epochs', type=int, default=300, help='Maximum epoch')
argparser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
argparser.add_argument('--wd', type=float, default=5e-6, help='Weight decay')
argparser.add_argument('--batch_size', type=int, default=1, help='batch size')
argparser.add_argument('--num_workers', type=int, default=0, help='number of workers')

# Data
argparser.add_argument('--dataset_root', type=str, default='/mnt/data/wzh/Datasets/AE2/AE2_data', help='dataset root')
argparser.add_argument('--dataset', type=str, default='tennis_forehand',
                       choices=['break_eggs', 'pour_milk', 'pour_liquid', 'tennis_forehand'],
                       help='AE2 dataset name')
argparser.add_argument('--view1', type=str, default='ego', help='view 1')
argparser.add_argument('--view2', type=str, default='exo', help='view 2 (can be same as view 1)')
argparser.add_argument('--input_size', type=int, default=224, help='frame input size: 168 or 224')
argparser.add_argument('--num_frames', type=int, default=32, help='number of frames: 20 or 32')
argparser.add_argument('--frame_stride', type=int, default=15, help='frame stride')
argparser.add_argument('--num_context_steps', type=int, default=2, help='context steps')
argparser.add_argument('--random_offset', type=int, default=1, help='random offset')

# Model
argparser.add_argument('--base_model_name', type=str, default='clip', help='Base model name')
argparser.add_argument('--vision_encoder_path', type=str,
                       default='/mnt/data/wzh/ai_model/openai-clip-vit-base-patch16',
                       help='Local CLIP ViT-B/16 or ViT-L/14 model directory')
argparser.add_argument('--freeze_base', action='store_true', help='whether to freeze base model')
argparser.add_argument('--num_tokens', type=int, default=196, help='token num in each frame (196 for clip ViT-B/16, 256 for clip ViT-L/14)')
argparser.add_argument('--hidden_dim', type=int, default=768, help='transformer hidden dim (768 for B/16 and 1024 for L/14)')
argparser.add_argument('--n_layers', type=int, default=12, help='transformer layer num')
argparser.add_argument('--n_layers_dec', type=int, default=4, help='transformer layer num')
argparser.add_argument('--n_heads', type=int, default=16, help='transformer heads num')
argparser.add_argument('--mlp_ratio', type=float, default=4., help='transformer mlp_ratio')
argparser.add_argument('--dp_rate', type=float, default=0.1, help='transformer dropout rate')
argparser.add_argument('--embedding_size', type=int, default=256, help='output embedding size')
argparser.add_argument('--decoder_embedding_size', type=int, default=256, help='output embedding size')
argparser.add_argument('--topk_ratio', type=float, default=0.3, help='token selection ratio')
argparser.add_argument('--mask_ratio', type=float, default=0.4, help='token masking ratio')

# Eval
argparser.add_argument('--ckpt', type=str, default='', help='model ckpt')
argparser.add_argument('--training_run', type=str, default='',
                       help='Standardized training run whose checkpoint and model config are used')
argparser.add_argument('--checkpoint_selection', type=str, default='val_loss',
                       choices=['val_loss', 'classification', 'retrieval', 'progression', 'kendall', 'last'],
                       help='Checkpoint selected from --training_run for embedding extraction')
argparser.add_argument('--extract_embedding', action='store_true', help='extract embeddings')
argparser.add_argument('--embedding_dir', type=str, default='',
                       help='Directory containing precomputed train/val or train/test .npy files')
argparser.add_argument('--embedding_file_split', type=str, default='', choices=['', 'val', 'test'],
                       help='Filename prefix for precomputed eval embeddings only; train_*.npy is always used by fitted tasks')
argparser.add_argument('--eval_task', type=str, default='1234', help='downstream evaluation')
argparser.add_argument('--eval_mode', type=str, default='test', choices=['val', 'test'],
                       help='Dataset split used for downstream evaluation')
argparser.add_argument('--device', type=str, default='auto', help='Evaluation device: auto, cpu, cuda, or cuda:N')
argparser.add_argument('--no_downstream_fit', action='store_true',
                       help='Disallow downstream SVM/linear-regression fitting; use eval_task 2 and/or 4')
argparser.add_argument('--no_probe_training', action='store_true', dest='no_downstream_fit',
                       help=argparse.SUPPRESS)
