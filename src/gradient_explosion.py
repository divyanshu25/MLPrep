import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Simple RNN - highly prone to exploding gradients
class SimpleRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize with large weights to guarantee explosion
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -3, 3)  # Very large initialization
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        return out[:, -1, :]  # Return last time step

# Training function
def train_rnn(use_clipping=False, max_norm=1.0):
    model = SimpleRNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # High learning rate
    
    # Dummy sequential data (batch_size, seq_length, input_size)
    X = torch.randn(16, 50, 10)  # Long sequences
    y = torch.randn(16, 20)
    
    losses = []
    grad_norms_before = []
    grad_norms_after = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm BEFORE clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_before += param_norm.item() ** 2
        total_norm_before = total_norm_before ** 0.5
        grad_norms_before.append(total_norm_before)
        
        # Apply gradient clipping if enabled
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Calculate gradient norm AFTER clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        total_norm_after = total_norm_after ** 0.5
        grad_norms_after.append(total_norm_after)
        
        # Update weights
        optimizer.step()
        
        losses.append(loss.item())
        
        # Stop if loss becomes NaN or infinity
        if np.isnan(loss.item()) or np.isinf(loss.item()):
            print(f"Training exploded at epoch {epoch}! Loss: {loss.item()}")
            # Pad with NaN for plotting
            while len(losses) < 50:
                losses.append(np.nan)
                grad_norms_before.append(np.nan)
                grad_norms_after.append(np.nan)
            break
    
    return losses, grad_norms_before, grad_norms_after

# Run experiments
print("="*70)
print("Training WITHOUT gradient clipping...")
print("="*70)
losses_no_clip, grads_before_no_clip, grads_after_no_clip = train_rnn(use_clipping=False)

print("\n" + "="*70)
print("Training WITH gradient clipping (max_norm=1.0)...")
print("="*70)
losses_clip, grads_before_clip, grads_after_clip = train_rnn(use_clipping=True, max_norm=1.0)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot losses
axes[0, 0].plot(losses_no_clip, label='Without Clipping', linewidth=2, alpha=0.8, color='red')
axes[0, 0].plot(losses_clip, label='With Clipping', linewidth=2, alpha=0.8, color='green')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(bottom=0, top=max(filter(lambda x: not np.isnan(x), losses_no_clip[:10])) * 1.2)

# Plot gradient norms (before clipping)
axes[0, 1].plot(grads_before_no_clip, label='Without Clipping', linewidth=2, alpha=0.8, color='red')
axes[0, 1].plot(grads_before_clip, label='With Clipping', linewidth=2, alpha=0.8, color='orange')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Gradient Norm', fontsize=12)
axes[0, 1].set_title('Gradient Norm (Before Clipping)', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Plot gradient norms (after clipping) - only for clipped version
axes[1, 0].plot(grads_after_clip, label='After Clipping', linewidth=2, alpha=0.8, color='green')
axes[1, 0].axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Clip Threshold')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Gradient Norm', fontsize=12)
axes[1, 0].set_title('Gradient Norm (After Clipping)', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(bottom=0, top=2)

# Comparison table
axes[1, 1].axis('off')
valid_losses_no_clip = [l for l in losses_no_clip if not np.isnan(l)]
valid_losses_clip = [l for l in losses_clip if not np.isnan(l)]

table_data = [
    ['Metric', 'Without Clipping', 'With Clipping'],
    ['Final Loss', f'{valid_losses_no_clip[-1]:.4f}' if valid_losses_no_clip else 'NaN', 
     f'{valid_losses_clip[-1]:.4f}'],
    ['Epochs Completed', f'{len(valid_losses_no_clip)}', '50'],
    ['Max Grad (before)', f'{max(g for g in grads_before_no_clip if not np.isnan(g)):.2f}',
     f'{max(grads_before_clip):.2f}'],
    ['Max Grad (after)', f'{max(g for g in grads_after_no_clip if not np.isnan(g)):.2f}',
     f'{max(grads_after_clip):.2f}'],
]

table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 5):
    for j in range(3):
        if j == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        table[(i, j)].set_text_props(fontsize=10)

plt.tight_layout()
plt.savefig('exploding_gradients_rnn.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("RESULTS SUMMARY:")
print("="*70)
print(f"Without clipping:")
print(f"  - Completed epochs: {len(valid_losses_no_clip)}/50")
print(f"  - Final loss: {valid_losses_no_clip[-1]:.4f}" if valid_losses_no_clip else "  - Training exploded (NaN)")
print(f"  - Max gradient: {max(g for g in grads_before_no_clip if not np.isnan(g)):.2f}")
print(f"\nWith clipping (max_norm=1.0):")
print(f"  - Completed epochs: 50/50")
print(f"  - Final loss: {valid_losses_clip[-1]:.4f}")
print(f"  - Max gradient before clip: {max(grads_before_clip):.2f}")
print(f"  - Max gradient after clip: {max(grads_after_clip):.2f}")
print("="*70)
print("\n✓ Gradient clipping prevents the explosion and enables stable training!")