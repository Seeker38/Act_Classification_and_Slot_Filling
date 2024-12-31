import torch
from torch.utils.data import Dataset
import json

class SlotFillingDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id=None, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        
        
        if self.label2id is None: # create label id
            # First, get all possible labels from the data
            unique_labels = {'O'}  # Initialize with 'O' tag
            with open(file_path, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    for turn in dialogue['dialogue']:
                        labels = turn.get('turn_label', [])
                        for slot_name, _ in labels:
                            unique_labels.add(f'B-{slot_name}')
                            unique_labels.add(f'I-{slot_name}')
            
            self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        self.data = self.load_and_process_data(file_path)
        
    def load_and_process_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        for dialogue in data:
            for turn in dialogue['dialogue']:
                text = turn['transcript']
                labels = turn.get('turn_label', [])
                
                encoding = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                
                
                label_ids = ['O'] * len(encoding['input_ids'][0]) # Init labels as 'O'
                
                
                for slot_name, slot_value in labels: # Map slots to BIO tags
                    # Tokenize slot value
                    slot_tokens = self.tokenizer.tokenize(slot_value)
                    text_tokens = self.tokenizer.tokenize(text)
                    
                    # Find slot tokens in text
                    for i in range(len(text_tokens)):
                        if text_tokens[i:i+len(slot_tokens)] == slot_tokens:
                            # Account for [CLS] token
                            label_ids[i+1] = f'B-{slot_name}'
                            for j in range(1, len(slot_tokens)):
                                if i+j+1 < len(label_ids):
                                    label_ids[i+j+1] = f'I-{slot_name}'
                
                label_ids = [self.label2id[label] for label in label_ids]
                labels_tensor = torch.tensor(label_ids)
                
                processed_data.append({
                    'input_ids': encoding['input_ids'][0],
                    'attention_mask': encoding['attention_mask'][0],
                    'labels': labels_tensor
                })
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]