"""
Improved Training script for Stage 2: Analog Clock Manipulator V2
With circular loss and better architecture
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import config
from src.models.analog_manipulator_v2 import create_analog_manipulator_v2, CircularMSELoss
from src.data.analog_dataset import get_analog_dataloaders
from src.utils.time_utils import sincos_to_angles, calculate_angle_error


class Stage2TrainerV2:
    """Improved trainer for Analog Clock Manipulator."""

    def __init__(self, model, dataloaders, device, checkpoint_dir):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use circular loss for angles
        self.criterion = CircularMSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=5e-5,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_error = float('inf')
        self.early_stopping_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_errors = []

    def calculate_angle_metrics(self, predictions, targets):
        """Calculate comprehensive angle metrics."""
        batch_size = predictions.size(0)
        errors = {
            'hour': [],
            'minute': [],
            'second': [],
            'mean': []
        }

        for i in range(batch_size):
            pred_angles = sincos_to_angles(predictions[i].cpu().numpy())
            true_angles = sincos_to_angles(targets[i].cpu().numpy())

            h_error = calculate_angle_error(pred_angles[0], true_angles[0])
            m_error = calculate_angle_error(pred_angles[1], true_angles[1])
            s_error = calculate_angle_error(pred_angles[2], true_angles[2])

            errors['hour'].append(h_error)
            errors['minute'].append(m_error)
            errors['second'].append(s_error)
            errors['mean'].append((h_error + m_error + s_error) / 3.0)

        return {
            'mean_hour': np.mean(errors['hour']),
            'mean_minute': np.mean(errors['minute']),
            'mean_second': np.mean(errors['second']),
            'mean_overall': np.mean(errors['mean']),
            'median_hour': np.median(errors['hour']),
            'median_minute': np.median(errors['minute']),
            'median_second': np.median(errors['second']),
            'max_hour': np.max(errors['hour']),
            'max_minute': np.max(errors['minute']),
            'max_second': np.max(errors['second'])
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [Train]")
        for images, times, sincos_angles in pbar:
            images = images.to(self.device)
            times = times.to(self.device)
            sincos_angles = sincos_angles.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images, times)
            loss = self.criterion(outputs, sincos_angles)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(sincos_angles.detach().cpu())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Scheduler step
        self.scheduler.step()

        avg_loss = running_loss / len(self.dataloaders['train'])
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        angle_metrics = self.calculate_angle_metrics(all_preds, all_targets)

        return avg_loss, angle_metrics

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc=f"Epoch {epoch} [Val]")
            for images, times, sincos_angles in pbar:
                images = images.to(self.device)
                times = times.to(self.device)
                sincos_angles = sincos_angles.to(self.device)

                outputs = self.model(images, times)
                loss = self.criterion(outputs, sincos_angles)

                running_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(sincos_angles.cpu())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.dataloaders['val'])
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        angle_metrics = self.calculate_angle_metrics(all_preds, all_targets)

        return avg_loss, angle_metrics

    def visualize_predictions(self, num_samples=5):
        """Visualize angle predictions."""
        self.model.eval()
        samples_visualized = 0

        with torch.no_grad():
            for images, times, sincos_angles in self.dataloaders['val']:
                if samples_visualized >= num_samples:
                    break

                images = images.to(self.device)
                times = times.to(self.device)
                outputs = self.model(images, times)

                for i in range(min(images.size(0), num_samples - samples_visualized)):
                    pred_angles = sincos_to_angles(outputs[i].cpu().numpy())
                    true_angles = sincos_to_angles(sincos_angles[i].numpy())

                    h_err = calculate_angle_error(pred_angles[0], true_angles[0])
                    m_err = calculate_angle_error(pred_angles[1], true_angles[1])
                    s_err = calculate_angle_error(pred_angles[2], true_angles[2])

                    print(f"Sample {samples_visualized + 1}:")
                    print(f"  Predicted: H={pred_angles[0]:.1f}° M={pred_angles[1]:.1f}° S={pred_angles[2]:.1f}°")
                    print(f"  True:      H={true_angles[0]:.1f}° M={true_angles[1]:.1f}° S={true_angles[2]:.1f}°")
                    print(f"  Errors:    H={h_err:.1f}° M={m_err:.1f}° S={s_err:.1f}°")

                    samples_visualized += 1

    def train(self, num_epochs):
        """Full training loop."""
        print("\n" + "="*70)
        print("STAGE 2 TRAINING V2: IMPROVED ANALOG CLOCK MANIPULATOR")
        print("="*70)

        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_errors.append(val_metrics['mean_overall'])

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Mean Error: {train_metrics['mean_overall']:.2f}°")
            print(f"  Val Loss:   {val_loss:.4f} | Mean Error: {val_metrics['mean_overall']:.2f}°")
            print(f"    Hour: {val_metrics['mean_hour']:.2f}° (max: {val_metrics['max_hour']:.1f}°)")
            print(f"    Minute: {val_metrics['mean_minute']:.2f}° (max: {val_metrics['max_minute']:.1f}°)")
            print(f"    Second: {val_metrics['mean_second']:.2f}° (max: {val_metrics['max_second']:.1f}°)")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  LR: {current_lr:.6f}")

            # Save best model based on angle error
            if val_metrics['mean_overall'] < self.best_val_error:
                self.best_val_error = val_metrics['mean_overall']
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model! Mean error: {self.best_val_error:.2f}°")
            else:
                self.early_stopping_counter += 1

            # Visualize predictions
            if epoch % 10 == 0:
                print("\n  Sample Predictions:")
                self.visualize_predictions(num_samples=5)

            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.early_stopping_counter >= 25:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break

        print("\n✅ Training complete!")
        print(f"Best validation error: {self.best_val_error:.2f}°")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Plot training curves
        self.plot_training_curves()

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Angle Error
        ax2.plot(self.val_errors, label='Val Mean Angle Error', color='red')
        ax2.axhline(y=5.0, color='green', linestyle='--', label='Target (<5°)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Angle Error (degrees)')
        ax2.set_title('Validation Angle Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Training curves saved to {plot_path}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_error': self.best_val_error,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        if is_best:
            path = self.checkpoint_dir / 'stage2_v2_best.pth'
        else:
            path = self.checkpoint_dir / f'stage2_v2_epoch_{epoch}.pth'

        torch.save(checkpoint, path)


def main():
    """Main training function."""
    print("Initializing Stage 2 V2 training...")

    # Update config for V2
    config.STAGE2_CONFIG['dropout_rate'] = 0.4
    config.STAGE2_CONFIG['time_embedding_dim'] = 64
    config.STAGE2_CONFIG['learning_rate'] = 5e-5

    # Create dataloaders
    dataloaders = get_analog_dataloaders(
        config.STAGE2_CONFIG,
        batch_size=config.STAGE2_CONFIG['batch_size'],
        num_workers=config.NUM_WORKERS
    )

    # Create improved model
    model = create_analog_manipulator_v2(config.STAGE2_CONFIG)
    print(f"\n✓ Model V2 created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = Stage2TrainerV2(
        model=model,
        dataloaders=dataloaders,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINTS_DIR / 'stage2_v2'
    )

    # Train
    trainer.train(num_epochs=config.STAGE2_CONFIG['epochs'])


if __name__ == '__main__':
    main()
