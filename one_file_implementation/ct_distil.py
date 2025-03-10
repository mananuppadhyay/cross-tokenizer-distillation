from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import json
import numpy as np
import pdb
device = torch.device("cuda:0")
save_dir = "/home/users/mananu/llm_things/distillation_op"
os.makedirs(save_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(save_dir, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    dataset = load_dataset("google-research-datasets/qed")
    if 'validation' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1)
        dataset['validation'] = dataset.pop('test')
    return dataset

def custom_collate_fn(tokenizer, model_name=""):
    def collate_fn(batch):
        full_texts = []
        
        is_phi_model = "phi" in model_name.lower()
        
        for item in batch:
            paragraph = item["paragraph_text"]
            question = item["question"]
            
            # Handle multiple answers
            answer_list = item['original_nq_answers']
            answers = [ans.get('string', '') for ans in answer_list if ans.get('string', '')]
            answer_part = "; ".join(answers) if answers else ""
            
            if is_phi_model:
                # Phi model prompt template
                system_message = "You are a helpful and precise question answering assistant."
                user_message = f"Answer the following question based on the given context:\n\nContext: {paragraph}\n\nQuestion: {question}"
                
                full_text = f"<|system|>system\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|><|assistant|>\n{answer_part}<|end|>"
            else:
                # Default format for other models
                input_part = f"{paragraph}\n{question}"
                full_text = f"<s>{input_part}</s><s>{answer_part}</s>" 
            
            full_texts.append(full_text)
        
        tokenized = tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
=        labels = tokenized["input_ids"].clone()
        
        for i, text in enumerate(full_texts):
            if is_phi_model:
                parts = text.split("<|assistant|>\n")
                if len(parts) > 1:
                    input_part = parts[0] + "<|assistant|>\n"
                    
                    input_tokens_len = len(tokenizer(input_part, add_special_tokens=False)['input_ids'])
                    
                    labels[i, :input_tokens_len] = -100
            else:
                parts = text.split("</s><s>")
                if len(parts) > 1:
                    input_part = parts[0] + "</s><s>"
                    
                    input_tokens_len = len(tokenizer(input_part, add_special_tokens=False)['input_ids'])
                    
                    labels[i, :input_tokens_len] = -100
        
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized["labels"] = labels
        return tokenized
    
    return collate_fn

dataset = load_and_preprocess_data()
teacher_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
student_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

te_collate_fn  = custom_collate_fn(teacher_tokenizer,"phi")
st_collate_fn = custom_collate_fn(student_tokenizer,"stud")
teacher_dataset = dataset

student_dataset = dataset


te_dataloader = DataLoader(
    teacher_dataset['train'],
    batch_size=1,
    collate_fn=te_collate_fn,
    shuffle=False
)

st_dataloader = DataLoader(
    student_dataset['train'],
    batch_size=1,
    collate_fn=st_collate_fn,
    shuffle=False
)
batch = next(iter(st_dataloader))


class DistillationModel(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()

    def forward(self, 
                student_input_ids: torch.Tensor, 
                student_attention_mask : torch.Tensor, 
                student_labels: torch.Tensor,
                teacher_input_ids: torch.Tensor, 
                teacher_attention_mask: torch.Tensor, 
                teacher_labels: torch.Tensor):
        
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_labels
        )
        # pdb.set_trace()
        return student_output, teacher_output


class DistillationLoss(nn.Module):
    def __init__(self, 
                 crossentropy_weight=1, 
                 distillation_weight=1.5, 
                 student_temperature=1, 
                 teacher_temperature=1, 
                 skip_student_eos=False, 
                 skip_teacher_eos=False, 
                 ignore_index=-100, 
                 tokenizer_student=None, 
                 tokenizer_teacher=None):
        
        super().__init__()
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.ignore_index = ignore_index


    def forward(self, 
                student_predictions, 
                teacher_predictions, 
                student_targets, 
                teacher_targets):
        
        student = student_predictions.logits
        teacher = teacher_predictions.logits

        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_targets)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_targets)

        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]

        for i in range(student.size(0)):
            shift = student_answer_index[i]
            size = student_answer_size[i]
            end_shift = shift+size
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            teacher[i] = torch.cat((
                torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
                torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )

        # Cut to max answer length
        mex_length = max(max(student_answer_size), max(teacher_answer_size))
        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]


        # Sort in descending order to align probabilities
        student = student.sort(dim=-1, descending=True).values
        teacher = teacher.sort(dim=-1, descending=True).values

        # Pad to get same vocabulary size
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)

        # Cross entropy loss
        crossentropy_loss = self.crossentropy_weight * student_predictions.loss

        distillation_loss = torch.zeros(student.size(0), device=student.device)
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            distillation_loss[i] = abs(student[i][:size] - teacher[i][:size]).sum(-1).mean(-1)
        distillation_loss = distillation_loss.mean()
        distillation_loss = self.distillation_weight * (distillation_loss)

        return crossentropy_loss + distillation_loss, crossentropy_loss, distillation_loss

    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size




def main():
    # Load models
    bnb_config  = BitsAndBytesConfig(
        load_in_4bit = True ,
        bnb_4bit_use_double_quant = True ,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_storage_dtype = torch.float16,
        )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda:0",
        # quantization_config= bnb_config,
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloomz-560m",
        device_map="cuda:0"
    )

    # Initialize components
    model = DistillationModel(student_model, teacher_model)
    optim = torch.optim.AdamW(model.parameters(), lr=9e-6)
    distillation_loss = DistillationLoss()

    # Training configuration
    num_epochs = 4
    best_val_loss = float('inf')
    loss_history = {
        'train_total': [],
        'train_ce': [],
        'train_distill': [],
        'val_total': [],
        'val_ce': [],
        'val_distill': []
    }

    # Load datasets and create dataloaders
    dataset = load_and_preprocess_data()
    # (Keep dataloader creation code here)
    # [Previous dataloader setup code remains the same]

    # Add validation dataloaders
    te_val_dataloader = DataLoader(
        teacher_dataset['validation'],
        batch_size=1,
        collate_fn=te_collate_fn,
        shuffle=False
    )
    st_val_dataloader = DataLoader(
        student_dataset['validation'],
        batch_size=1,
        collate_fn=st_collate_fn,
        shuffle=False
    )

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(zip(st_dataloader, te_dataloader)), 
                    total=len(st_dataloader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (st_batch, te_batch) in progress_bar:
            optim.zero_grad()
            
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
            loss, cross_loss, dist_loss = distillation_loss(st_output, te_output, student_labels, teacher_labels)
            loss.backward()
            optim.step()
            
            # Update loss history
            loss_history['train_total'].append(loss.item())
            loss_history['train_ce'].append(cross_loss.item())
            loss_history['train_distill'].append(dist_loss.item())
            if(i%100==0):
                    logger.info(f"Total Loss: {loss_history['train_total'][-1]:.4f} | CE Loss: {loss_history['train_ce'][-1]:.4f} | Distill Loss: {loss_history['train_distill'][-1]:.4f}")

        torch.save(student_model.state_dict(), 
             os.path.join(save_dir, f"student_model{epoch+1}.pt"))
        # Validation loop
        model.eval()
        val_loss, val_ce, val_distill = 0, 0, 0
        with torch.no_grad():
            for st_batch, te_batch in zip(st_val_dataloader, te_val_dataloader):
                # Forward pass and loss calculation
                # (Similar to training step but without backward pass)
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
                loss, ce_loss, distill_loss = distillation_loss(st_output, te_output, 
                                                              student_labels, teacher_labels)
                val_loss += loss.item()
                val_ce += ce_loss.item()
                val_distill += distill_loss.item()

        # Calculate averages
        avg_train_loss = np.mean(loss_history['train_total'][-len(st_dataloader):])
        avg_val_loss = val_loss / len(st_val_dataloader)
        loss_history['val_total'].append(avg_val_loss)
        loss_history['val_ce'].append(val_ce / len(st_val_dataloader))
        loss_history['val_distill'].append(val_distill / len(st_val_dataloader))

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student_model.state_dict(), 
                      os.path.join(save_dir, "best_student_model.pt"))
            logger.info(f"New best model saved with val loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'student_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'loss_history': loss_history,
        }
        # torch.save(checkpoint, 
        #          os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))

        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"CE Loss: {loss_history['val_ce'][-1]:.4f} | Distill Loss: {loss_history['val_distill'][-1]:.4f}")

    # Final save
    torch.save(student_model.state_dict(), 
             os.path.join(save_dir, "final_student_model.pt"))
    student_tokenizer.save_pretrained(save_dir)
    
    # Save loss history
    with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)
    
    # Plotting
    plot_training_curves(loss_history, save_dir)

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

if __name__ == "__main__":
    main()