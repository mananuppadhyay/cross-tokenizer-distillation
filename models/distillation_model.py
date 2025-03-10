import torch
import torch.nn as nn

class DistillationModel(nn.Module):
    """
    Model that combines a student and teacher model for knowledge distillation.
    
    Args:
        student: Student model to be trained
        teacher: Teacher model used as reference
    """
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()  # Set teacher to evaluation mode

    def forward(self, 
                student_input_ids: torch.Tensor, 
                student_attention_mask: torch.Tensor, 
                student_labels: torch.Tensor,
                teacher_input_ids: torch.Tensor, 
                teacher_attention_mask: torch.Tensor, 
                teacher_labels: torch.Tensor):
        """
        Forward pass through both student and teacher models.
        
        Args:
            student_input_ids: Input IDs for student model
            student_attention_mask: Attention mask for student model
            student_labels: Labels for student model
            teacher_input_ids: Input IDs for teacher model
            teacher_attention_mask: Attention mask for teacher model
            teacher_labels: Labels for teacher model
            
        Returns:
            tuple: Student and teacher model outputs
        """
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
        
        return student_output, teacher_output