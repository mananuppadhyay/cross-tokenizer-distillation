import os
import torch
import json
from evaluate import load
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pdb
def load_and_preprocess_data():
    dataset = load_dataset("google-research-datasets/qed")
    if 'validation' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1)
        dataset['validation'] = dataset.pop('test')
    return dataset

def evaluate_model(model, tokenizer, dataset, device, max_new_tokens=50, save_path="generations_phi4_mini_instruct.json"):
    """
    Evaluate the model on the validation set and calculate precision, recall, and F1 score
    """
    squad_metric = load("squad")
    
    model.eval()
    predictions = []
    references = []
    results_list = []
    
    # Process validation dataset
    for idx in tqdm(range(len(dataset['validation'])), desc="Evaluating"):
        example = dataset['validation'][idx]
        system_message = "You are a helpful and precise question answering assistant."
        user_message = f"{example['paragraph_text']}\n{example['question']}"
        
        # Format input using the specified prompt format
        full_text = f"<|system|>\n{system_message}<|end|><|user|>\n{user_message}<|end|><|assistant|>\n"
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True
            )
        
        answer_start = inputs.input_ids.shape[-1]
        decoded_output = tokenizer.decode(outputs[0][answer_start:], skip_special_tokens=True).strip()        # Prepare ground truth answers
        ground_truths = [a['string'] for a in example['original_nq_answers'] if a['string']]
        
        predictions.append({'prediction_text': decoded_output, 'id': str(idx)})
        references.append({
            'answers': {
                'text': ground_truths,
                'answer_start': [0]*len(ground_truths) 
            },
            'id': str(idx)
        })
        
        results_list.append({
            'id': idx,
            'paragraph': example['paragraph_text'],
            'question': example['question'],
            'generated_answer': decoded_output,
            'ground_truth_answers': ground_truths
        })
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4)
    
    results = squad_metric.compute(predictions=predictions, references=references)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred['prediction_text'].lower().split())
        ref_tokens = [set(gt.lower().split()) for gt in ref['answers']['text']]
        
        best_tp = 0
        best_fp = 0
        best_fn = 0
        
        for gt_tokens in ref_tokens:
            tp = len(pred_tokens & gt_tokens)
            fp = len(pred_tokens - gt_tokens)
            fn = len(gt_tokens - pred_tokens)
            
            if tp > best_tp:
                best_tp = tp
                best_fp = fp
                best_fn = fn
                
        true_positives += best_tp
        false_positives += best_fp
        false_negatives += best_fn
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'squad_em': results['exact_match'],
        'squad_f1': results['f1'],
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }

# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    dataset = load_and_preprocess_data()
    
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        save_path=""
    )
    
    print("\nEvaluation Results:")
    print(f"SQuAD EM: {metrics['squad_em']:.4f}")
    print(f"SQuAD F1: {metrics['squad_f1']:.4f}")
    print(f"Token Precision: {metrics['token_precision']:.4f}")
    print(f"Token Recall: {metrics['token_recall']:.4f}")
    print(f"Token F1: {metrics['token_f1']:.4f}")