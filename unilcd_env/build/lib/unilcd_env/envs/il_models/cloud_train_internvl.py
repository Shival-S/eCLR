"""
Training script for InternVL3-2B Cloud Model with LoRA fine-tuning.
Uses Parameter-Efficient Fine-Tuning (PEFT) to train on CARLA dataset.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path

from cloud_model_internvl import CloudModelInternVL, CloudModelInternVLLite
from cloud_dataloader import get_dataloader

# Optional: LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. Run: pip install peft")


def setup_lora(model, lora_r=16, lora_alpha=32):
    """
    Apply LoRA adapters to the model for efficient fine-tuning.
    Only trains ~0.1% of parameters while keeping base model frozen.
    """
    if not PEFT_AVAILABLE:
        print("PEFT not available. Training full model (requires more VRAM)")
        return model, None

    # LoRA configuration for vision model
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    # Apply LoRA to vision model
    model.vision_model = get_peft_model(model.vision_model, lora_config)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, lora_config


def train(data_folder, save_path, use_lite=True, use_lora=True, epochs=50, batch_size=8):
    """
    Train InternVL cloud model on CARLA dataset.

    Args:
        data_folder: Path to CARLA dataset with .jpg and .npy files
        save_path: Path to save trained model
        use_lite: Use lite version (vision encoder only) for faster training
        use_lora: Use LoRA for efficient fine-tuning
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    start_time = time.time()

    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing InternVL Cloud Model")
    print("=" * 60)

    if use_lite:
        print("Using Lite version (vision encoder only)")
        model = CloudModelInternVLLite(device=device)
    else:
        print("Using full InternVL model")
        model = CloudModelInternVL(device=device)

    # Apply LoRA if requested
    lora_config = None
    if use_lora:
        model, lora_config = setup_lora(model)

    model.to(device)

    # Unfreeze action head and goal network (always trainable)
    for param in model.action_head.parameters():
        param.requires_grad = True
    for param in model.goal.parameters():
        param.requires_grad = True

    # Optimizer - only train unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    criterion = nn.L1Loss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Data loader
    print(f"\nLoading data from: {data_folder}")
    train_loader = get_dataloader(data_folder, batch_size)
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training for {epochs} epochs")
    print("=" * 60)

    loss_values = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # batch = (images, actions, locations)
            images = batch[0].to(device)
            actions_gt = batch[1].to(device)
            locations = batch[2].to(device)

            # Forward pass
            actions_pred = model(images, locations)

            # Compute loss
            loss = criterion(actions_pred, actions_gt)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Progress update every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.6f}")

        # Epoch statistics
        epoch_loss = total_loss / num_batches
        loss_values.append(epoch_loss)
        scheduler.step()

        time_elapsed = time.time() - start_time
        time_per_epoch = time_elapsed / (epoch + 1)
        eta = time_per_epoch * (epochs - epoch - 1)

        print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {epoch_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | ETA: {eta:.0f}s")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch, save_path, lora_config)
            print(f"  -> Saved best model (loss: {best_loss:.6f})")

    # Final save
    final_path = save_path.replace('.pth', '_final.pth')
    save_checkpoint(model, optimizer, epochs, final_path, lora_config)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title('InternVL Cloud Model Training Loss')
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace('.pth', '_learning_curve.jpg'))
    plt.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)


def save_checkpoint(model, optimizer, epoch, path, lora_config=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lora_config': lora_config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InternVL Cloud Model Training with LoRA')
    parser.add_argument('-d', '--data_folder', default="../../data/",
                        type=str, help='Path to CARLA dataset')
    parser.add_argument('-s', '--save_path', default="./cloud_model_internvl.pth",
                        type=str, help='Path to save model')
    parser.add_argument('--lite', action='store_true', default=True,
                        help='Use lite version (vision encoder only)')
    parser.add_argument('--no-lora', action='store_true',
                        help='Disable LoRA (train full model)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank')

    args = parser.parse_args()

    train(
        data_folder=args.data_folder,
        save_path=args.save_path,
        use_lite=args.lite,
        use_lora=not args.no_lora,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
