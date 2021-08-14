# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import gzip
import json
import torch
import pandas as pd
import random
from datasets import tqdm


class SyntheticQuestionsMultiDatasets(torch.utils.data.Dataset):
    """
    Dataset  reader for Synthetic Quetions IIRC
    """

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_input_token_len,
                 generation_model=False,
                 num_examples_to_load=None,
                 max_output_token_len=None):

        # load lot data from gzip
        self.generation_model = generation_model

        random.seed(42)
        examples = []
        with gzip.open(data_path, "r") as f:
            for i, l in enumerate(tqdm(f)):
                examples.append(json.loads(l))
                #
                # if i == 100:
                #     break

        # sample num_examples_to_load examples
        if num_examples_to_load is not None:
            num_examples_to_load = min(len(examples), num_examples_to_load)
            examples = random.sample(examples, num_examples_to_load)


        self.data = pd.DataFrame([[example['qid'],
                                   example['question'],
                                   example['context'],
                                   example['answer'],
                                   example['template']]
                                  for example in examples], columns=['qids', 'phrases', 'contexts', 'gold', 'type'])

        self.lengths = [len(self.data.contexts[k]) + len(self.data.phrases[k])
                        for k in range(len(self.data))]

        self.qids = self.data.qids
        self.phrases = self.data.phrases
        self.contexts = self.data.contexts
        self.gold = self.data.gold
        self.types = self.data.type

        self.tokenizer = tokenizer
        self.max_input_token_len = max_input_token_len
        self.max_output_token_len = max_output_token_len

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, index):

        phrase = str(self.phrases[index])
        context = str(self.contexts[index])
        type = str(self.types[index])
        qid = str(self.qids[index])

        source_text = phrase + '\n' + context
        source_text = 'QA: ' + source_text

        gold_text = self.gold[index]

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
            'answer_type': type,
            'id': qid,
            'question': phrase,
            'context': context,
            'answer': gold_text
        }
