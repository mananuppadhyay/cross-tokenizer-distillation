import os
import torch
import logging
import argparse
from datetime import datetime

from data.dataset import load_and_preprocess_data
from data.dataloader import create_dataloaders
from models.distillation_model import DistillationModel
from models.losses import DistillationLoss
from training.trainer import train_and_evaluate
from utils.plotting import plot_training_curves
from utils.config import ModelConfig

def setup_logging(save_dir):
    logging.basicConfig(
        filename=os.path.join(save_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Distillation Training")
    parser.add_argument("--save_dir", type=str, default="./distillation_output", 
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--teacher_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Teacher model name or path")
    parser.add_argument("--student_model", type=str, default="bigscience/bloomz-560m", 
                        help="Student model name or path")
    parser.add_argument("--num_epochs", type=int, default=4, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=9e-6, 
                        help="Learning rate")
    parser.add_argument("--ce_weight", type=float, default=1.0, 
                        help="Cross-entropy loss weight")
    parser.add_argument("--distill_weight", type=float, default=1.5, 
                        help="Distillation loss weight")
    parser.add_argument("--load_4bit", type=bool, default=True, 
                        help="Whether to load models in 4-bit precision")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="Logging interval during training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logging(save_dir)
    logger.info(f"Starting distillation with arguments: {args}")
    
    config = ModelConfig(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        load_in_4bit=args.load_4bit,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        ce_weight=args.ce_weight,
        distill_weight=args.distill_weight,
        save_dir=save_dir,
        log_interval=args.log_interval
    )
    
    # Load dataset
    dataset = load_and_preprocess_data()
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        dataset, 
        config.teacher_model_name, 
        config.student_model_name, 
        config.batch_size
    )
    
    distillation_model, optimizer, loss_fn = initialize_training_components(config)
    
    loss_history = train_and_evaluate(
        model=distillation_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataloaders=dataloaders,
        config=config,
        logger=logger
    )
    
    dataloaders["student_tokenizer"].save_pretrained(save_dir)
    torch.save(distillation_model.student.state_dict(), 
             os.path.join(save_dir, "final_student_model.pt"))
    
    plot_training_curves(loss_history, save_dir)
    
    logger.info("Training completed successfully")

def initialize_training_components(config):
    from models.model_utils import load_models
    
    teacher_model, student_model, teacher_tokenizer, student_tokenizer = load_models(config)
    
    distillation_model = DistillationModel(student_model, teacher_model)
    
    optimizer = torch.optim.AdamW(
        distillation_model.parameters(), 
        lr=config.learning_rate
    )
    
    loss_fn = DistillationLoss(
        crossentropy_weight=config.ce_weight,
        distillation_weight=config.distill_weight
    )
    
    return distillation_model, optimizer, loss_fn

if __name__ == "__main__":
    main()