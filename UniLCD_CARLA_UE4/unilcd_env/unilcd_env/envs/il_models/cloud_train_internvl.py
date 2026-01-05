"""
Training script for InternVL3 cloud model.

Supports:
- Frozen vision encoder + trainable action head
- LoRA fine-tuning of vision encoder
- Mixed precision training (bfloat16)
- Checkpointing and resume
- Validation split
- TensorBoard logging

Usage:
    python cloud_train_internvl.py \
        -d /data4/shival/UniLCD_CARLA_UE4/data \
        -s cloud_model_internvl.pth \
        --epochs 50 \
        --batch_size 4 \
        --use_lora
"""

import torch
import torch.nn as nn
import argparse
import time
import os
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

from cloud_model_internvl import InternVLCloudModel, InternVLCloudModelLite
from cloud_dataloader_internvl import get_internvl_dataloader, get_train_val_dataloaders


def train(
    data_folder: str,
    save_path: str,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    use_lora: bool = True,
    freeze_vision: bool = True,
    use_lite_model: bool = False,
    val_split: float = 0.1,
    checkpoint_freq: int = 10,
    resume_from: str = None,
    use_tensorboard: bool = True
):
    """
    Train InternVL3 cloud model on CARLA driving data.

    Args:
        data_folder: Path to training data
        save_path: Path to save final model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        use_lora: Whether to use LoRA for vision encoder fine-tuning
        freeze_vision: Whether to freeze vision encoder (ignored if use_lora=True)
        use_lite_model: Use lighter InternViT-300M instead of full InternVL3-8B
        val_split: Fraction of data for validation
        checkpoint_freq: Save checkpoint every N epochs
        resume_from: Path to checkpoint to resume from
        use_tensorboard: Whether to log to TensorBoard
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create checkpoint directory
    checkpoint_dir = Path(save_path).parent / "checkpoints_internvl"
    checkpoint_dir.mkdir(exist_ok=True)

    # TensorBoard setup
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(save_path).parent / "logs_internvl" / datetime.now().strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        except ImportError:
            print("TensorBoard not available, skipping logging")
            use_tensorboard = False
            writer = None
    else:
        writer = None

    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing model...")
    print("=" * 60)

    if use_lite_model:
        print("Using lite model (InternViT-300M only)")
        model = InternVLCloudModelLite(freeze_vision=freeze_vision)
    else:
        print("Using full InternVL3-8B")
        model = InternVLCloudModel(
            freeze_vision=freeze_vision,
            use_lora=use_lora,
            lora_r=16,
            pooling="mean"
        )

    model = model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    # Data loaders
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    if val_split > 0:
        train_loader, val_loader = get_train_val_dataloaders(
            data_folder, batch_size, num_workers=4, val_split=val_split, augment=True
        )
    else:
        train_loader = get_internvl_dataloader(
            data_folder, batch_size, num_workers=4, augment=True
        )
        val_loader = None

    # Optimizer - only train parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Resume optimizer state
    if resume_from and os.path.exists(resume_from):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - start_epoch, eta_min=learning_rate * 0.01
    )

    # Loss function
    criterion = nn.L1Loss()

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\n" + "=" * 60)
    print(f"Starting training for {epochs - start_epoch} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA: {use_lora}")
    print(f"  Validation split: {val_split}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch_idx, (images, actions, locations) in enumerate(train_loader):
            images = images.to(device)
            actions = actions.to(device)
            locations = locations.to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = model(images, locations)
                loss = criterion(predictions.float(), actions)

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            num_batches += 1

            # Log batch progress
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f}")

        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for images, actions, locations in val_loader:
                    images = images.to(device)
                    actions = actions.to(device)
                    locations = locations.to(device)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        predictions = model(images, locations)
                        loss = criterion(predictions.float(), actions)

                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            val_losses.append(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = str(Path(save_path).with_suffix('')) + "_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved! Val loss: {avg_val_loss:.6f}")
        else:
            avg_val_loss = None

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Timing
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch - start_epoch + 1)) * (epochs - epoch - 1) if epoch > start_epoch else 0

        # Log to console
        val_str = f"Val Loss: {avg_val_loss:.6f} | " if avg_val_loss else ""
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | {val_str}"
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            if avg_val_loss:
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {
                    'use_lora': use_lora,
                    'freeze_vision': freeze_vision,
                    'use_lite_model': use_lite_model,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final model
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to: {save_path}")

    # Save training history
    history_path = str(Path(save_path).with_suffix('')) + "_history.json"
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_lora': use_lora,
            'freeze_vision': freeze_vision,
            'use_lite_model': use_lite_model
        }
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('InternVL Cloud Model Training')
    plt.legend()
    plt.grid(True)
    plot_path = str(Path(save_path).with_suffix('')) + "_learning_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to: {plot_path}")

    # Print summary
    total_time = time.time() - start_time
    print(f"\nTraining Summary:")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"  Final val loss: {val_losses[-1]:.6f}")
        print(f"  Best val loss: {best_val_loss:.6f}")

    if writer:
        writer.close()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train InternVL3 Cloud Model')
    parser.add_argument('-d', '--data_folder', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('-s', '--save_path', type=str, default="cloud_model_internvl.pth",
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Use LoRA for vision encoder fine-tuning')
    parser.add_argument('--freeze_vision', action='store_true', default=True,
                        help='Freeze vision encoder weights')
    parser.add_argument('--use_lite', action='store_true', default=False,
                        help='Use lighter InternViT-300M model')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable TensorBoard logging')

    args = parser.parse_args()

    train(
        data_folder=args.data_folder,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        freeze_vision=args.freeze_vision,
        use_lite_model=args.use_lite,
        val_split=args.val_split,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume,
        use_tensorboard=not args.no_tensorboard
    )
