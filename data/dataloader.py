import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def create_dataloaders(dataset, teacher_model_name, student_model_name, batch_size=1):
    """
    Create dataloaders for training and validation.
    
    Args:
        dataset: The dataset with train and validation splits
        teacher_model_name: Name of the teacher model
        student_model_name: Name of the student model
        batch_size: Batch size for training
        
    Returns:
        dict: Dictionary containing dataloaders and tokenizers
    """
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    teacher_collate_fn = custom_collate_fn(teacher_tokenizer, model_name=teacher_model_name)
    student_collate_fn = custom_collate_fn(student_tokenizer, model_name=student_model_name)
    
    teacher_train_dataloader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        collate_fn=teacher_collate_fn,
        shuffle=False
    )
    
    student_train_dataloader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        collate_fn=student_collate_fn,
        shuffle=False
    )
    
    teacher_val_dataloader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        collate_fn=teacher_collate_fn,
        shuffle=False
    )
    
    student_val_dataloader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        collate_fn=student_collate_fn,
        shuffle=False
    )
    
    return {
        "teacher_train": teacher_train_dataloader,
        "student_train": student_train_dataloader,
        "teacher_val": teacher_val_dataloader,
        "student_val": student_val_dataloader,
        "teacher_tokenizer": teacher_tokenizer,
        "student_tokenizer": student_tokenizer
    }

def custom_collate_fn(tokenizer, model_name=""):
    """
    Custom collate function for batching data.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        model_name: Name of the model to format inputs for
        
    Returns:
        function: Collate function for DataLoader
    """
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
        
        labels = tokenized["input_ids"].clone()
        
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