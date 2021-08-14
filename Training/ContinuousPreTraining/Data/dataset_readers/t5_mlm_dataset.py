# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import json
import random

import torch
from datasets import tqdm


class T5MlmDataset(torch.utils.data.Dataset):

    def __init__(self, data_path,
                 tokenizer,
                 max_input_token_len,
                 max_output_token_len,
                 num_examples_to_load=1000,
                 num_wiki_examples=7529903):

        # init
        self.num_examples_to_load = num_examples_to_load
        self.tokenizer = tokenizer
        self.max_input_token_len = max_input_token_len
        self.max_output_token_len = max_output_token_len
        epoch_indices = random.sample(range(0, num_wiki_examples), self.num_examples_to_load)
        epoch_indices = set(epoch_indices)

        self.wiki_prefix = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('Wiki: '))
        self.wiki_examples = []

        with open(data_path, "r") as f:
            for i, l in tqdm(enumerate(f)):
                if i in epoch_indices:
                    self.wiki_examples.append(json.loads(l))

                if i == num_wiki_examples:
                    break

    def __len__(self):
        return len(self.wiki_examples)

    def __getitem__(self, index):

        # if wiki source, add pad indices for short sequences and return
        num_input_pads = self.max_input_token_len - len(self.wiki_prefix) - len(self.wiki_examples[index]['inputs'])
        num_label_pads = self.max_output_token_len - len(self.wiki_examples[index]['labels'])
        input_ids_tensor = torch.IntTensor(self.wiki_prefix
                                           + self.wiki_examples[index]['inputs']
                                           + [self.tokenizer.pad_token_id] * num_input_pads).to(dtype=torch.long)
        labels_tensor = torch.IntTensor(self.wiki_examples[index]['labels']
                                        + [-100] * num_label_pads).to(dtype=torch.long)

        # create attention masks
        attention_mask = torch.ones(input_ids_tensor.shape)
        attention_mask[input_ids_tensor == self.tokenizer.pad_token_id] = 0

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask.to(dtype=torch.long),
            'labels': labels_tensor,
        }
