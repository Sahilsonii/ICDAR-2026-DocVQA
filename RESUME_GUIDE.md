# Training Resume Guide

This guide explains how to resume training if it crashes, gets interrupted, or you want to continue from a previous checkpoint.

## Overview

The training script now supports three ways to resume:

1. **Auto-detect mode** (`--resume-auto`): Automatically finds and resumes from the latest checkpoint
2. **Explicit checkpoint** (`--resume-checkpoint <PATH>`): Resume from a specific checkpoint
3. **W&B run** (`--resume-run-id <ID>`): Resume a W&B run from the cloud

## Quick Start

### Method 1: Auto-Resume (Easiest)

If training crashes, just run:

```bash
python scripts/train_all_gpu.py --resume-auto --epochs 50
```

The script will automatically detect the latest checkpoint and resume from there.

### Method 2: Using the Helper Script

For convenience, use the provided helper script:

```bash
# View checkpoint and W&B info
python scripts/resume_training.py

# Auto-resume training
python scripts/resume_training.py --resume --epochs 50

# Generate the resume command
python scripts/resume_training.py --generate-command
```

### Method 3: Explicit Checkpoint Path

If you know the checkpoint path:

```bash
python scripts/train_all_gpu.py --resume-checkpoint checkpoints/qwen-7b/checkpoint-50 --epochs 50
```

### Method 4: Resume W&B Run

To resume a specific W&B run:

```bash
python scripts/train_all_gpu.py --resume-run-id xdtedi7j --epochs 50
```

Find your run ID at [wandb.ai/docvqa2026/docvqa2026](https://wandb.ai/docvqa2026/docvqa2026) or in the `wandb/` folder.

## Where to Find Checkpoint & Run Info

### Checkpoint Locations
- Checkpoints are saved in: `checkpoints/<model>/checkpoint-<step>/`
- Example: `checkpoints/qwen-7b/checkpoint-50/`

### W&B Run ID
- View at: https://wandb.ai/docvqa2026/docvqa2026/runs
- Local path: `wandb/run-<timestamp>-<runid>/`
- Extract ID from folder name: `run-20260325_135839-**l607q83m**`

## Script Behavior

### How Resume Works

1. **From Checkpoint**:
   - Loads model weights from checkpoint
   - Resumes optimizer state
   - Continues from the last training step
   - Uses the same W&B run if available

2. **From W&B Run**:
   - Resumes the same W&B run
   - Auto-detects latest checkpoint if available
   - Continues logging metrics to the same dashboard

### Automatic Checkpoint Detection

If you don't specify `--resume-checkpoint` or `--resume-run-id`, the trainer will:
1. Look in the output directory for existing checkpoints
2. Find the latest checkpoint (by step number)
3. Automatically resume from there
4. Log information about the resumed checkpoint

## Command Examples

### Resume with Different Epochs
```bash
# Train for more epochs
python scripts/train_all_gpu.py --resume-auto --epochs 100
```

### Resume with Different Learning Rate
```bash
python scripts/train_all_gpu.py --resume-checkpoint checkpoints/qwen-7b/checkpoint-30 --lr 1e-4 --epochs 50
```

### Resume with Different Model
Note: You can't switch models mid-training. The adapter is tied to the base model weights.

## Monitoring Training

During training, you can:
- **View real-time metrics**: https://wandb.ai/docvqa2026/docvqa2026
- **Check terminal logs**: They show current epoch, step, and loss
- **View GPU usage**: `nvidia-smi` or `watch -n 1 nvidia-smi`

## Common Questions

### Q: How do I know if training was interrupted?
A: Check if new checkpoints were created in `checkpoints/<model>/`. If the latest checkpoint is recent, training may have crashed.

### Q: Can I resume a crashed run to a different machine?
A: Yes! Use `--resume-run-id <ID>` to continue on another machine. The checkpoint weights will be downloaded automatically.

### Q: What happens if I resume with different `--epochs`?
A: It will train for the specified number of epochs from the checkpoint. This is useful if you want to train longer than originally planned.

### Q: Will resuming lose progress?
A: No! Resuming loads:
- Model weights from the checkpoint
- Optimizer state (momentum, etc.)
- Training metrics to W&B
- You continue exactly where you left off

### Q: Can I resume from an old checkpoint?
A: Yes, but only if the model hasn't been modified. You can resume from any checkpoint in `checkpoints/<model>/checkpoint-*`.

## Troubleshooting

### Issue: "No checkpoints found"
- Check if training has started: look for `checkpoints/` directory
- Verify the model key matches: `--model qwen-7b` (not qwen-3b)
- Check file permissions in `checkpoints/` directory

### Issue: "CUDA Out of Memory" when resuming
- Clear GPU memory: `python -c "import torch; torch.cuda.empty_cache()"`
- Reduce batch size: `--batch-size 1` (already default)
- Reduce number of epochs

### Issue: W&B run not syncing during resume
- Check internet connection
- Verify `WANDB_API_KEY` is set: `echo $WANDB_API_KEY`
- Try: `wandb login` to re-authenticate

## File Structure

```
ICDAR-2026-DocVQA/
├── checkpoints/
│   └── qwen-7b/
│       ├── checkpoint-50/      # Resume from here
│       ├── checkpoint-100/
│       └── final_adapter/
├── wandb/
│   └── run-20260325_135839-l607q83m/  # Run ID: l607q83m
├── scripts/
│   ├── train_all_gpu.py        # Main training script
│   └── resume_training.py      # Helper script
└── ...
```

## Advanced: Manual Resumption

If you need to manually set training parameters:

```bash
# Resume with custom settings
python scripts/train_all_gpu.py \
  --model qwen-7b \
  --resume-checkpoint checkpoints/qwen-7b/checkpoint-50 \
  --epochs 100 \
  --lr 2e-4 \
  --batch-size 1
```

---

**Need help?** Check the W&B dashboard at https://wandb.ai/docvqa2026/docvqa2026 for detailed training metrics and logs.
