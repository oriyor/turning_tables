# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import json
import torch
import gzip
import pandas as pd
import hashlib
from datasets import tqdm
from ContinuousPreTraining.Common.config import Config
from torch.utils.data.dataset import Dataset
import random


class UnifiedQaDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_input_token_len,
                 max_output_token_len=32,
                 prefix=None,
                 sample_size=None,
                 generation_model=False
                 ):

        # load lot data from gzip
        random.seed(42)
        self.generation_model = generation_model
        examples = []

        self.prefix = 'QA: ' if prefix is None else prefix

        with gzip.open(data_path, "r") as f:
            for i, l in enumerate(tqdm(f)):
                examples.append(json.loads(l))

        # sample num_examples_to_load examples
        if sample_size is not None:
            sample_size = min(len(examples), sample_size)
            examples = random.sample(examples, sample_size)

        for example in examples:
            if 'id' not in example:
                m = hashlib.md5()
                m.update(example['context'].encode())
                m.update(example['answer'].encode())
                example['id'] = m.hexdigest()

        self.data = pd.DataFrame([[example['context'],
                                   example['answer'],
                                   example['all_answers'],
                                   example['id']]
                                  for example in examples],
                                 columns=['contexts', 'gold', 'all_answers', 'id'])

        self.tokenizer = tokenizer
        self.max_input_token_len = max_input_token_len
        self.max_output_token_len = max_output_token_len

        # lengths for fast grouping
        self.lengths = [len(context)
                        for context in self.data.contexts]

    def __len__(self):
        return len(self.data.id)

    def __getitem__(self, index):

        source_text = str(self.data.contexts[index])

        if self.generation_model:
            #if Config().get('train_dataset_reader.datasets.unifiedqa.add_prefix'):
            source_text = self.prefix + source_text
            gold_text = self.data.gold[index]
            labels = self.tokenizer.batch_encode_plus([gold_text], max_length=self.max_output_token_len,
                                                      truncation=True, padding='max_length',
                                                      return_tensors='pt').input_ids.squeeze() \
                .to(dtype=torch.long)
            labels[labels == 0] = -100
            tokenized_inputs = self.tokenizer.encode_plus(text=source_text,
                                                          add_special_tokens=True,
                                                          max_length=self.max_input_token_len,
                                                          pad_to_max_length=False,
                                                          return_token_type_ids=False,
                                                          return_attention_mask=True,
                                                          return_overflowing_tokens=False,
                                                          return_special_tokens_mask=False,
                                                          )
            return {
                'input_ids': tokenized_inputs.input_ids,
                'attention_mask': tokenized_inputs.attention_mask,
                'labels': labels,
                'question': source_text,
                'answer': gold_text,
                'id': self.data.id[index],
                'all_answers': json.dumps(self.data.all_answers[index])
            }

        else:
            tokenized = self.tokenizer.batch_encode_plus([source_text], max_length=self.max_input_token_len,
                                                         truncation=True, padding='max_length',
                                                         return_tensors='pt')
            return {
                'input_ids': tokenized.input_ids.squeeze().to(dtype=torch.long),
                'attention_mask': tokenized.attention_mask,
                'labels': torch.tensor(self.data.gold[index])
            }
