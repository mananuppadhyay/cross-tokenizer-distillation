class ModelConfig:
    """
    Configuration class to store model and training parameters.
    
    Args:
        teacher_model_name: Name or path of teacher model
        student_model_name: Name or path of student model
        device: Device to use for training
        load_in_4bit: Whether to load models in 4-bit precision
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        ce_weight: Cross-entropy loss weight
        distill_weight: Distillation loss weight
        save_dir: Directory to save outputs
        log_interval: Interval for logging during training
    """
    def __init__(
        self,
        teacher_model_name="microsoft/Phi-3-mini-4k-instruct",
        student_model_name="bigscience/bloomz-560m",
        device="cuda:0",
        load_in_4bit=True,
        learning_rate=9e-6,
        num_epochs=4,
        batch_size=1,
        ce_weight=1.0,
        distill_weight=1.5,
        save_dir="./distillation_output",
        log_interval=100
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.ce_weight = ce_weight
        self.distill_weight = distill_weight
        self.save_dir = save_dir
        self.log_interval = log_interval