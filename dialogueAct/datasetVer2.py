import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

class DialogueActDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.texts = [x for x, _ in samples]
        
        self.acts = []
        for _, y in samples:
            if isinstance(y, dict):
                self.acts.append(list(y.keys()))
            elif isinstance(y, str):
                self.acts.append([y])
            else:
                raise ValueError(f"Unexpected label type: {type(y)}. Expected dict or str.")
        
        self.label_binarizer = MultiLabelBinarizer()
        self.labels = self.label_binarizer.fit_transform(self.acts)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }
