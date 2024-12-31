import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import json

class JointDialogueDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id=None, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        
        # For dialogue acts
        self.act_binarizer = MultiLabelBinarizer()
        
        if self.label2id is None:
            unique_labels = {'O'}
            with open(file_path, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    for turn in dialogue['dialogue']:
                        labels = turn.get('turn_label', [])
                        for slot_name, _ in labels:
                            unique_labels.add(f'B-{slot_name}')
                            unique_labels.add(f'I-{slot_name}')
            
            self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.data = self.load_and_process_data(file_path)
        
    def load_and_process_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # First pass: collect all dialogue acts for binarizer fitting
        all_acts = []
        for dialogue in data:
            for turn in dialogue['dialogue']:
                acts = turn.get('dialogue_act', None)
                if isinstance(acts, dict):
                    all_acts.append(list(acts.keys()))
                elif isinstance(acts, str):
                    all_acts.append([acts])
                else:
                    all_acts.append([])
    
        self.act_binarizer.fit(all_acts)
        
        # Second pass: process the data
        processed_data = []
        for dialogue_idx, dialogue in enumerate(data):
            for turn in dialogue['dialogue']:
                text = turn['transcript']
                slot_labels = turn.get('turn_label', [])
                
                
                acts = turn.get('dialogue_act', None) # Process dialogue acts
                if isinstance(acts, dict):
                    act_labels = list(acts.keys())
                elif isinstance(acts, str):
                    act_labels = [acts]
                else:
                    act_labels = []
                act_binary = self.act_binarizer.transform([act_labels])[0]
                
                tokens = self.tokenizer.tokenize(text)
                
                token_labels = ['O'] * len(tokens)
                
                
                for slot_name, slot_value in slot_labels: # Map slots to BIO tags
                    slot_tokens = self.tokenizer.tokenize(slot_value)
                    
                    # Find slot tokens in text
                    for i in range(len(tokens)):
                        if tokens[i:i+len(slot_tokens)] == slot_tokens:
                            token_labels[i] = f'B-{slot_name}'
                            for j in range(1, len(slot_tokens)):
                                if i+j < len(token_labels):
                                    token_labels[i+j] = f'I-{slot_name}'
                
                encoding = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                
                label_ids = ['O']  # [CLS]
                label_ids.extend(token_labels)
                label_ids.append('O')  # [SEP]
                
                # Pad to max length
                if len(label_ids) < self.max_len:
                    label_ids.extend(['O'] * (self.max_len - len(label_ids)))
                else:
                    label_ids = label_ids[:self.max_len]
                
                # Convert string labels to IDs
                label_ids = [self.label2id[label] for label in label_ids]
                labels_tensor = torch.tensor(label_ids)
                
                processed_data.append({
                    'input_ids': encoding['input_ids'][0],
                    'attention_mask': encoding['attention_mask'][0],
                    'act_labels': torch.tensor(act_binary, dtype=torch.float),
                    'slot_labels': labels_tensor
                })
        
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]