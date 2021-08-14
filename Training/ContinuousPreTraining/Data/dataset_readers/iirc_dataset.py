# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import json
import torch
import pandas as pd


class IircDataset(torch.utils.data.Dataset):
    """
    Dataset  reader for IIRC
    """

    def __init__(self, data_path, tokenizer, max_seq_len,
                 prefix=None, generation_model=False, summ_len=None):

        # load lot data from gzip
        self.generation_model = generation_model
        self.prefix = 'QA: ' if prefix is None else prefix

        with open(data_path, "r") as f:
            data = json.load(f)

        examples = []
        question_index = 0
        for context in data:
            for q in context['questions']:
                example = {'qid': q['qid'] if 'qid' in q else str(question_index),
                           'phrase': q['question'],
                           'context': self.preprocess_iirc_context(q['context']),
                           'answer': self.preprocess_iirc_answer(q['answer']),
                           'answer_type': q['answer']['type']}
                examples.append(example)
                question_index += 1

        self.data = pd.DataFrame([[example['qid'],
                                   example['phrase'],
                                   example['context'],
                                   example['answer'],
                                   example['answer_type']]
                                  for example in examples], columns=['qids', 'phrases', 'contexts', 'gold', 'type'])

        self.qids = self.data.qids
        self.phrases = self.data.phrases
        self.contexts = self.data.contexts
        self.gold = self.data.gold
        self.types = self.data.type

        self.tokenizer = tokenizer
        self.source_len = max_seq_len
        self.summ_len = summ_len

    def preprocess_iirc_context(self, context):
        """
        preprocess a context from iirc dataset to a text, which can be used by a generative model
        """
        return '\n'.join([c['passage'] + ': ' + c['text'] for c in context])

    def preprocess_iirc_answer(self, answer):
        """
        preprocess an answer from iirc dataset to a text, which can be used by a generative model
        """
        if answer['type'] == 'none':
            return 'none'
        if answer['type'] == 'span':
            return '#'.join([a['text'] for a in answer['answer_spans']])
        if answer['type'] in ['binary', 'value']:
            return answer['answer_value']
        return None

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, index):

        phrase = str(self.phrases[index])
        context = str(self.contexts[index])
        type = str(self.types[index])
        qid = str(self.qids[index])

        source_text = phrase + '\n' + context
        source_text = self.prefix + source_text

        gold_text = self.gold[index]

        labels = self.tokenizer.batch_encode_plus([gold_text], max_length=self.summ_len,
                                                  truncation=True, padding='max_length',
                                                  return_tensors='pt').input_ids.squeeze() \
            .to(dtype=torch.long)
        labels[labels == 0] = -100
        tokenized_inputs = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len,
                                                            truncation=True, padding='max_length',
                                                            return_tensors='pt')

        return {
            'input_ids': tokenized_inputs.input_ids.squeeze().to(dtype=torch.long),
            'attention_mask': tokenized_inputs.attention_mask,
            'labels': labels,
            'answer_type': type,
            'id': qid,
            'question': phrase,
            'context': context,
            'answer': gold_text,
            'all_answers': json.dumps(gold_text.split('#'))
        }
