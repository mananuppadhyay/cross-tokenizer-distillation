from datasets import load_dataset

def load_and_preprocess_data():
    """
    Load and preprocess the QED dataset.
    
    Returns:
        dataset: Processed dataset with train and validation splits
    """
    dataset = load_dataset("google-research-datasets/qed")
    
    # Create validation split if not present
    if 'validation' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1)
        dataset['validation'] = dataset.pop('test')
        
    return dataset