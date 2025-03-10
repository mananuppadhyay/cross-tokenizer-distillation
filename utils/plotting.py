import os
import matplotlib.pyplot as plt


def plot_training_curves(loss_history, save_dir):
    plt.figure(figsize=(15, 10))
    
    # Training losses
    plt.subplot(2, 1, 1)
    plt.plot(loss_history['train_total'], label='Total Loss')
    plt.plot(loss_history['train_ce'], label='CE Loss')
    plt.plot(loss_history['train_distill'], label='Distill Loss')
    plt.title('Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation losses
    plt.subplot(2, 1, 2)
    plt.plot(loss_history['val_total'], label='Total Loss')
    plt.plot(loss_history['val_ce'], label='CE Loss')
    plt.plot(loss_history['val_distill'], label='Distill Loss')
    plt.title('Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()