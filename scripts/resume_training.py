#!/usr/bin/env python3
"""
Resume Training Helper Script
=============================
This script helps you resume training if it crashes or gets interrupted.
It provides an easy way to:
1. Show the latest checkpoint path
2. Show the latest W&B run ID
3. Auto-generate the resume command
4. Resume training with a single command
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_latest_checkpoint(model_key):
    """Get the latest checkpoint for a model."""
    checkpoint_dir = Path("checkpoints") / model_key
    if not checkpoint_dir.exists():
        return None, None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None, None
    
    latest = checkpoints[-1]
    return latest, latest.name


def get_latest_wandb_run(project="docvqa2026"):
    """Get the latest W&B run ID from wandb/ folder."""
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        return None
    
    runs = sorted(wandb_dir.glob("run-*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        return None
    
    latest_run = runs[0]
    run_name = latest_run.name
    # Extract run ID from "run-20260325_135839-l607q83m"
    run_id = run_name.split("-")[-1]
    
    return run_id, run_name


def format_resume_command(model_key="qwen-7b", checkpoint=None, run_id=None):
    """Generate a resume command."""
    cmd = f"python scripts/train_all_gpu.py --model {model_key}"
    
    if checkpoint:
        cmd += f" --resume-checkpoint {checkpoint}"
    elif run_id:
        cmd += f" --resume-run-id {run_id}"
    else:
        cmd += " --resume-auto"
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Resume training helper")
    parser.add_argument("--model", default="qwen-7b", help="Model key (e.g., qwen-7b)")
    parser.add_argument("--show-checkpoint", action="store_true", help="Show latest checkpoint")
    parser.add_argument("--show-wandb-id", action="store_true", help="Show latest W&B run ID")
    parser.add_argument("--generate-command", action="store_true", help="Generate resume command")
    parser.add_argument("--resume", action="store_true", help="Resume training automatically")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint to resume from")
    parser.add_argument("--run-id", default=None, help="Specific W&B run ID to resume")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (for --resume)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  📚 ICDAR 2026 DocVQA — Training Resume Helper")
    print("="*70 + "\n")
    
    checkpoint, checkpoint_name = get_latest_checkpoint(args.model)
    run_id, run_name = get_latest_wandb_run()
    
    # Show checkpoint info
    if args.show_checkpoint or not any([args.show_wandb_id, args.generate_command, args.resume]):
        print("📂 Latest Checkpoint Information:")
        if checkpoint:
            print(f"   Path: {checkpoint}")
            print(f"   Name: {checkpoint_name}")
        else:
            print("   ⚠️  No checkpoints found for this model")
    
    # Show W&B run ID
    if args.show_wandb_id or not any([args.show_checkpoint, args.generate_command, args.resume]):
        print("\n🔍 Latest W&B Run:")
        if run_id:
            print(f"   Run ID: {run_id}")
            print(f"   Path:   {run_name}")
            print(f"   Dashboard: https://wandb.ai/docvqa2026/docvqa2026/runs/{run_id}")
        else:
            print("   ⚠️  No W&B runs found")
    
    # Generate command
    if args.generate_command or args.resume:
        cmd = format_resume_command(
            model_key=args.model,
            checkpoint=args.checkpoint or str(checkpoint),
            run_id=args.run_id or run_id
        )
        
        if args.resume_run_id or args.resume:
            cmd += f" --epochs {args.epochs}"
        
        if args.generate_command:
            print("\n📋 Resume Command:")
            print(f"   {cmd}")
        
        if args.resume:
            print("\n▶️  Starting training resume...\n")
            os.system(cmd)
    
    print("\n" + "="*70)
    print("\n💡 USAGE EXAMPLES:\n")
    print("   # View latest checkpoint and W&B run:")
    print("   python scripts/resume_training.py\n")
    print("   # Show only checkpoint:")
    print("   python scripts/resume_training.py --show-checkpoint\n")
    print("   # Show only W&B run ID:")
    print("   python scripts/resume_training.py --show-wandb-id\n")
    print("   # Generate resume command:")
    print("   python scripts/resume_training.py --generate-command\n")
    print("   # Auto-resume from latest checkpoint:")
    print("   python scripts/resume_training.py --resume\n")
    print("   # Resume from specific checkpoint:")
    print("   python scripts/resume_training.py --resume --checkpoint checkpoints/qwen-7b/checkpoint-100\n")
    print("   # Resume W&B run:")
    print("   python scripts/resume_training.py --resume --run-id xdtedi7j\n")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
