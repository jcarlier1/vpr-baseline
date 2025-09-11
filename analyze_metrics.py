#!/usr/bin/env python3
"""
Analyze and plot training metrics from CSV logs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_training_metrics(log_file):
    """Load and parse training metrics from CSV"""
    df = pd.read_csv(log_file)
    
    # Separate different types of events
    batch_data = df[df['event_type'] == 'batch'].copy()
    epoch_data = df[df['event_type'] == 'epoch_summary'].copy()
    validation_data = df[df['event_type'] == 'validation'].copy()
    
    # Convert numeric columns
    for col in ['total_loss', 'contrastive_loss', 'triplet_loss', 'learning_rate', 'validation_recall']:
        if col in batch_data.columns:
            batch_data[col] = pd.to_numeric(batch_data[col], errors='coerce')
            epoch_data[col] = pd.to_numeric(epoch_data[col], errors='coerce')
            validation_data[col] = pd.to_numeric(validation_data[col], errors='coerce')
    
    return batch_data, epoch_data, validation_data

def plot_training_metrics(log_file, save_plots=True):
    """Create comprehensive training plots"""
    batch_data, epoch_data, validation_data = load_training_metrics(log_file)
    
    model_name = Path(log_file).stem
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Metrics: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Batch-level losses over time
    ax1 = axes[0, 0]
    if len(batch_data) > 0:
        # Create a continuous x-axis using batch number within epoch
        batch_data['global_batch'] = batch_data['epoch'] * batch_data['batch'].max() + batch_data['batch']
        
        ax1.plot(batch_data['global_batch'], batch_data['total_loss'], label='Total Loss', alpha=0.7)
        ax1.plot(batch_data['global_batch'], batch_data['contrastive_loss'], label='Contrastive Loss', alpha=0.7)
        ax1.plot(batch_data['global_batch'], batch_data['triplet_loss'], label='Triplet Loss', alpha=0.7)
        ax1.set_xlabel('Global Batch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Batch-Level Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Epoch-level losses
    ax2 = axes[0, 1]
    if len(epoch_data) > 0:
        ax2.plot(epoch_data['epoch'], epoch_data['total_loss'], 'o-', label='Total Loss', linewidth=2, markersize=6)
        ax2.plot(epoch_data['epoch'], epoch_data['contrastive_loss'], 's-', label='Contrastive Loss', linewidth=2, markersize=6)
        ax2.plot(epoch_data['epoch'], epoch_data['triplet_loss'], '^-', label='Triplet Loss', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Epoch Summary Losses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Validation recall
    ax3 = axes[0, 2]
    if len(validation_data) > 0:
        ax3.plot(validation_data['epoch'], validation_data['validation_recall'], 'ro-', linewidth=3, markersize=8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall@1')
        ax3.set_title('Validation Recall@1')
        ax3.grid(True, alpha=0.3)
        
        # Add best recall annotation
        best_idx = validation_data['validation_recall'].idxmax()
        best_epoch = validation_data.loc[best_idx, 'epoch']
        best_recall = validation_data.loc[best_idx, 'validation_recall']
        ax3.annotate(f'Best: {best_recall:.4f}', 
                    xy=(best_epoch, best_recall), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 4. Learning rate schedule
    ax4 = axes[1, 0]
    if len(epoch_data) > 0:
        ax4.plot(epoch_data['epoch'], epoch_data['learning_rate'], 'g-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    # 5. Loss distribution (histogram)
    ax5 = axes[1, 1]
    if len(batch_data) > 0:
        ax5.hist(batch_data['total_loss'].dropna(), bins=30, alpha=0.7, label='Total', density=True)
        ax5.hist(batch_data['contrastive_loss'].dropna(), bins=30, alpha=0.7, label='Contrastive', density=True)
        ax5.hist(batch_data['triplet_loss'].dropna(), bins=30, alpha=0.7, label='Triplet', density=True)
        ax5.set_xlabel('Loss Value')
        ax5.set_ylabel('Density')
        ax5.set_title('Loss Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Training progress summary
    ax6 = axes[1, 2]
    if len(validation_data) > 0 and len(epoch_data) > 0:
        # Dual y-axis plot
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(epoch_data['epoch'], epoch_data['total_loss'], 'b-', label='Total Loss', linewidth=2)
        line2 = ax6_twin.plot(validation_data['epoch'], validation_data['validation_recall'], 'r-', label='Validation R@1', linewidth=2)
        
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss', color='b')
        ax6_twin.set_ylabel('Recall@1', color='r')
        ax6.set_title('Loss vs Validation Performance')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='center right')
        
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = log_file.replace('.csv', '_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_file}")
    
    plt.show()

def analyze_all_logs(log_dir="logs"):
    """Analyze all training logs in directory"""
    log_files = glob.glob(os.path.join(log_dir, "*.csv"))
    
    if not log_files:
        print(f"No CSV files found in {log_dir}")
        return
    
    print(f"Found {len(log_files)} training logs:")
    
    summary_data = []
    
    for log_file in sorted(log_files):
        print(f"\n--- Analyzing {os.path.basename(log_file)} ---")
        
        try:
            batch_data, epoch_data, validation_data = load_training_metrics(log_file)
            
            # Extract summary statistics
            model_name = Path(log_file).stem
            
            if len(validation_data) > 0:
                best_recall = validation_data['validation_recall'].max()
                final_recall = validation_data['validation_recall'].iloc[-1]
            else:
                best_recall = final_recall = 0.0
            
            if len(epoch_data) > 0:
                final_loss = epoch_data['total_loss'].iloc[-1]
                initial_loss = epoch_data['total_loss'].iloc[0]
                loss_reduction = (initial_loss - final_loss) / initial_loss * 100
            else:
                final_loss = initial_loss = loss_reduction = 0.0
            
            summary_data.append({
                'model': model_name,
                'best_recall': best_recall,
                'final_recall': final_recall,
                'final_loss': final_loss,
                'loss_reduction_%': loss_reduction,
                'total_epochs': len(epoch_data)
            })
            
            print(f"Best Recall@1: {best_recall:.4f}")
            print(f"Final Recall@1: {final_recall:.4f}")
            print(f"Loss Reduction: {loss_reduction:.1f}%")
            
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")
    
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(log_dir, "training_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n=== TRAINING SUMMARY ===")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze VPR training metrics')
    parser.add_argument('--log_file', type=str, help='Specific log file to analyze')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing log files')
    parser.add_argument('--no_plots', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    if args.log_file:
        # Analyze specific file
        if not args.no_plots:
            plot_training_metrics(args.log_file)
        
        batch_data, epoch_data, validation_data = load_training_metrics(args.log_file)
        print(f"\nMetrics loaded:")
        print(f"  Batch records: {len(batch_data)}")
        print(f"  Epoch records: {len(epoch_data)}")
        print(f"  Validation records: {len(validation_data)}")
        
    else:
        # Analyze all files in directory
        analyze_all_logs(args.log_dir)

if __name__ == "__main__":
    main()
