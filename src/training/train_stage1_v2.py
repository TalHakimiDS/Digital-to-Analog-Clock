"""
Improved Training script for Stage 1: Digital Clock Reader V2
With better architecture, loss, and training strategies
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
from src.models.digital_reader_v2 import create_digital_reader_v2, DigitCrossEntropyLoss
from src.data.digital_dataset import get_digital_dataloaders


class Stage1TrainerV2:
    """Improved trainer for Digital Clock Reader."""

    def __init__(self, model, dataloaders, device, checkpoint_dir):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Digit-wise cross entropy over HH:MM:SS digits
        self.criterion = DigitCrossEntropyLoss()

        # Optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=5e-5,  # Lower LR for better convergence
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=1e-6
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _time_tensor_to_hms(self, time_tensor: torch.Tensor):
        """
        Convert normalized time tensor (batch, 3) to integer (h, m, s).
        """
        h = torch.round(time_tensor[:, 0] * 23).clamp(0, 23).long()
        m = torch.round(time_tensor[:, 1] * 59).clamp(0, 59).long()
        s = torch.round(time_tensor[:, 2] * 59).clamp(0, 59).long()
        return h, m, s

    def _hms_to_digits(self, h: torch.Tensor, m: torch.Tensor, s: torch.Tensor):
        """
        Convert hour/minute/second tensors to 6 digit tensors (batch, 6).
        Positions: [H1, H2, M1, M2, S1, S2].
        """
        h1 = (h // 10).clamp(0, 2)
        h2 = (h % 10).clamp(0, 9)
        m1 = (m // 10).clamp(0, 5)
        m2 = (m % 10).clamp(0, 9)
        s1 = (s // 10).clamp(0, 5)
        s2 = (s % 10).clamp(0, 9)
        return torch.stack([h1, h2, m1, m2, s1, s2], dim=1)

    def _logits_to_hms(self, logits: torch.Tensor):
        """
        Convert digit logits (batch, 6, 10) to integer (h, m, s).
        """
        digits = logits.argmax(dim=2)  # (batch, 6)
        h = (digits[:, 0] * 10 + digits[:, 1]).clamp(0, 23)
        m = (digits[:, 2] * 10 + digits[:, 3]).clamp(0, 59)
        s = (digits[:, 4] * 10 + digits[:, 5]).clamp(0, 59)
        return h, m, s

    def calculate_accuracy(self, pred_logits, target_times):
        """Calculate exact match and per-component accuracy."""
        exact_match = 0
        hour_correct = 0
        minute_correct = 0
        second_correct = 0
        total = pred_logits.size(0)

        pred_h, pred_m, pred_s = self._logits_to_hms(pred_logits)
        true_h, true_m, true_s = self._time_tensor_to_hms(target_times)

        hour_correct = (pred_h == true_h).sum().item()
        minute_correct = (pred_m == true_m).sum().item()
        second_correct = (pred_s == true_s).sum().item()
        exact_match = ((pred_h == true_h) &
                       (pred_m == true_m) &
                       (pred_s == true_s)).sum().item()

        return {
            'exact_match': 100.0 * exact_match / total,
            'hour': 100.0 * hour_correct / total,
            'minute': 100.0 * minute_correct / total,
            'second': 100.0 * second_correct / total
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_logits = []
        all_times = []

        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [Train]")
        for images, times in pbar:
            images = images.to(self.device)
            times = times.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(images)

            # Build digit targets from normalized times
            with torch.no_grad():
                h, m, s = self._time_tensor_to_hms(times)
                digit_targets = self._hms_to_digits(h, m, s)

            loss = self.criterion(logits, digit_targets.to(self.device))

            # Backward
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_times.append(times.detach().cpu())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Scheduler step
        self.scheduler.step()

        avg_loss = running_loss / len(self.dataloaders['train'])
        all_logits = torch.cat(all_logits)
        all_times = torch.cat(all_times)
        accuracy = self.calculate_accuracy(all_logits, all_times)

        return avg_loss, accuracy

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_logits = []
        all_times = []

        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc=f"Epoch {epoch} [Val]")
            for images, times in pbar:
                images = images.to(self.device)
                times = times.to(self.device)

                logits = self.model(images)

                with torch.no_grad():
                    h, m, s = self._time_tensor_to_hms(times)
                    digit_targets = self._hms_to_digits(h, m, s)

                loss = self.criterion(logits, digit_targets.to(self.device))

                running_loss += loss.item()
                all_logits.append(logits.cpu())
                all_times.append(times.cpu())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.dataloaders['val'])
        all_logits = torch.cat(all_logits)
        all_times = torch.cat(all_times)
        accuracy = self.calculate_accuracy(all_logits, all_times)

        return avg_loss, accuracy

    def visualize_predictions(self, num_samples=8):
        """Visualize some predictions for debugging."""
        self.model.eval()
        samples_visualized = 0

        with torch.no_grad():
            for images, times in self.dataloaders['val']:
                if samples_visualized >= num_samples:
                    break

                images = images.to(self.device)
                times = times.to(self.device)

                logits = self.model(images)
                pred_h, pred_m, pred_s = self._logits_to_hms(logits.cpu())
                true_h, true_m, true_s = self._time_tensor_to_hms(times.cpu())

                for i in range(min(images.size(0), num_samples - samples_visualized)):
                    ph, pm, ps = int(pred_h[i]), int(pred_m[i]), int(pred_s[i])
                    th, tm, ts = int(true_h[i]), int(true_m[i]), int(true_s[i])

                    print(f"Sample {samples_visualized + 1}:")
                    print(f"  Predicted: {ph:02d}:{pm:02d}:{ps:02d}")
                    print(f"  True:      {th:02d}:{tm:02d}:{ts:02d}")
                    print(f"  Match: {ph == th and pm == tm and ps == ts}")

                    samples_visualized += 1

    def train(self, num_epochs):
        """Full training loop."""
        print("\n" + "="*70)
        print("STAGE 1 TRAINING V2: IMPROVED DIGITAL CLOCK READER")
        print("="*70)

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc['exact_match'])
            self.val_accs.append(val_acc['exact_match'])

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Exact Match: {train_acc['exact_match']:.2f}%")
            print(f"    Hour: {train_acc['hour']:.2f}% | Min: {train_acc['minute']:.2f}% | Sec: {train_acc['second']:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Exact Match: {val_acc['exact_match']:.2f}%")
            print(f"    Hour: {val_acc['hour']:.2f}% | Min: {val_acc['minute']:.2f}% | Sec: {val_acc['second']:.2f}%")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  LR: {current_lr:.6f}")

            # Save best model based on accuracy
            if val_acc['exact_match'] > self.best_val_acc:
                self.best_val_acc = val_acc['exact_match']
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model! Accuracy: {self.best_val_acc:.2f}%")
            else:
                self.early_stopping_counter += 1

            # Visualize predictions every 10 epochs
            if epoch % 10 == 0:
                print("\n  Sample Predictions:")
                self.visualize_predictions(num_samples=5)

            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.early_stopping_counter >= 25:  # Increased patience
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break

        print("\n✅ Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
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

        # Accuracy
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Exact Match Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
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
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        if is_best:
            path = self.checkpoint_dir / 'stage1_v2_best.pth'
        else:
            path = self.checkpoint_dir / f'stage1_v2_epoch_{epoch}.pth'

        torch.save(checkpoint, path)


def main():
    """Main training function."""
    print("Initializing Stage 1 V2 training...")

    # Update config for V2
    config.STAGE1_CONFIG['dropout_rate'] = 0.5
    config.STAGE1_CONFIG['learning_rate'] = 5e-5

    # Create dataloaders
    dataloaders = get_digital_dataloaders(
        config.STAGE1_CONFIG,
        batch_size=config.STAGE1_CONFIG['batch_size'],
        num_workers=config.NUM_WORKERS
    )

    # Create improved model
    model = create_digital_reader_v2(config.STAGE1_CONFIG)
    print(f"\n✓ Model V2 created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = Stage1TrainerV2(
        model=model,
        dataloaders=dataloaders,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINTS_DIR / 'stage1_v2'
    )

    # Train
    trainer.train(num_epochs=config.STAGE1_CONFIG['epochs'])


if __name__ == '__main__':
    main()
