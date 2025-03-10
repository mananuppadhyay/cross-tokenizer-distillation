import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_models(config):
    """
    Load teacher and student models with appropriate configurations.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        tuple: Teacher model, student model, teacher tokenizer, student tokenizer
    """
    # Configure quantization if needed
    bnb_config = None
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_storage_dtype=torch.float16,
        )
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        device_map=config.device,
        quantization_config=bnb_config if config.load_in_4bit else None,
        trust_remote_code=True
    )
    
    # Load student model (no quantization for student during training)
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        device_map=config.device
    )
    
    # Load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    
    return teacher_model, student_model, teacher_tokenizer, student_tokenizer