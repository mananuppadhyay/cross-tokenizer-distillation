import os
import torch
import numpy as np
from tqdm import tqdm
import json

def train_and_evaluate(model, optimizer, loss_fn, dataloaders, config, logger):
    """
    Train and evaluate the distillation model.
    
    Args:
        model: DistillationModel instance
        optimizer: Optimizer
        loss_fn: Loss function
        dataloaders: Dictionary of dataloaders
        config: Training configuration
        logger: Logger instance
        
    Returns:
        dict: Loss history
    """
    # Initialize tracking variables
    best_val_loss = float('inf')
    loss_history = {
        'train_total': [],
        'train_ce': [],
        'train_distill': [],
        'val_total': [],
        'val_ce': [],
        'val_distill': []
    }
    
    device = config.device
    num_epochs = config.num_epochs
    log_interval = config.log_interval
    save_dir = config.save_dir
    
    # Get dataloaders
    student_train_dataloader = dataloaders["student_train"]
    teacher_train_dataloader = dataloaders["teacher_train"]
    student_val_dataloader = dataloaders["student_val"]
    teacher_val_dataloader = dataloaders["teacher_val"]
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(
            enumerate(zip(student_train_dataloader, teacher_train_dataloader)), 
            total=len(student_train_dataloader), 
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        
        for i, (st_batch, te_batch) in progress_bar:
            optimizer.zero_grad()
            
            # Move data to device
            student_input_ids = st_batch["input_ids"].to(device)
            student_attention_mask = st_batch["attention_mask"].to(device)
            student_labels = st_batch["labels"].to(device)
            teacher_input_ids = te_batch["input_ids"].to(device)
            teacher_attention_mask = te_batch["attention_mask"].to(device)
            teacher_labels = te_batch["labels"].to(device) 
            
            # Forward pass
            st_output, te_output = model(
                student_input_ids,
                student_attention_mask,
                student_labels,
                teacher_input_ids,
                teacher_attention_mask,
                teacher_labels
            )       
        
            # Calculate loss
            loss, cross_loss, dist_loss = loss_fn(st_output, te_output, student_labels, teacher_labels)
            loss.backward()
            optimizer.step()
            
            # Update loss history
            loss_history['train_total'].append(loss.item())
            loss_history['train_ce'].append(cross_loss.item())
            loss_history['train_distill'].append(dist_loss.item())
            
            # Log progress
            if i % log_interval == 0:
                logger.info(f"Epoch {epoch+1}, Iteration {i}/{len(student_train_dataloader)}")
                logger.info(f"Total Loss: {loss_history['train_total'][-1]:.4f} | CE Loss: {loss_history['train_ce'][-1]:.4f} | Distill Loss: {loss_history['train_distill'][-1]:.4f}")
        
        # Save model checkpoint after each epoch
        torch.save(model.student.state_dict(), 
                 os.path.join(save_dir, f"student_model_epoch_{epoch+1}.pt"))
        
        # Validation phase
        model.eval()
        val_loss, val_ce, val_distill = 0, 0, 0
        num_val_batches = len(student_val_dataloader)
        
        with torch.no_grad():
            for st_batch, te_batch in zip(student_val_dataloader, teacher_val_dataloader):
                # Move data to device
                student_input_ids = st_batch["input_ids"].to(device)
                student_attention_mask = st_batch["attention_mask"].to(device)
                student_labels = st_batch["labels"].to(device)
                teacher_input_ids = te_batch["input_ids"].to(device)
                teacher_attention_mask = te_batch["attention_mask"].to(device)
                teacher_labels = te_batch["labels"].to(device) 
                
                # Forward pass
                st_output, te_output = model(
                    student_input_ids,
                    student_attention_mask,
                    student_labels,
                    teacher_input_ids,
                    teacher_attention_mask,
                    teacher_labels
                )
                
                # Calculate loss
                loss, ce_loss, distill_loss = loss_fn(st_output, te_output, 
                                                  student_labels, teacher_labels)
                val_loss += loss.item()
                val_ce += ce_loss.item()
                val_distill += distill_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = val_loss / num_val_batches
        avg_val_ce = val_ce / num_val_batches
        avg_val_distill = val_distill / num_val_batches
        
        # Update validation loss history
        loss_history['val_total'].append(avg_val_loss)
        loss_history['val_ce'].append(avg_val_ce)
        loss_history['val_distill'].append(avg_val_distill)
        
        # Calculate average training loss
        avg_train_loss = np.mean(loss_history['train_total'][-len(student_train_dataloader):])
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.student.state_dict(), 
                      os.path.join(save_dir, "best_student_model.pt"))
            logger.info(f"New best model saved with val loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'student_state_dict': model.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'loss_history': loss_history,
        }
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"CE Loss: {avg_val_ce:.4f} | Distill Loss: {avg_val_distill:.4f}")
        
        # Save current loss history
        with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f)
    
    return loss_history