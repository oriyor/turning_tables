# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import json
import torch
import pandas as pd
from transformers import T5Tokenizer
import matplotlib.pyplot as plt

from ContinuousPreTraining.Common.file_utils import cached_path


class IircRetrievalDataset(torch.utils.data.Dataset):
    """
    Dataset  reader for IIRC
    """
    def __init__(self, data_path, tokenizer, max_seq_len, retrieval_file=None,
                 generation_model=False, summ_len=None):

        # load lot data from gzip
        self.generation_model = generation_model
        self.retrieval_file = retrieval_file
        self.preprorcess_context_method = self.preprocess_iirc_context_with_retrieval if retrieval_file else self.preprocess_iirc_context_train

        with open(data_path, "r") as f:
            data = json.load(f)

        # read retrieval contexts
        if retrieval_file:
            self.retrieved_contexts = []
            json_file_path = cached_path(retrieval_file)
            with open(json_file_path) as json_file:
                for l in json_file:
                    question_retrieval_data = json.loads(l)
                    retrieved_sentences = [{'text': self.preprocess_retrieved_sentece(retrieved_context['sent']),
                                            'passage': retrieved_context['title']} for retrieved_context in
                                           question_retrieval_data['context_retrieval']['predicted_link_name_sent_list']]
                    self.retrieved_contexts.append(retrieved_sentences)

        examples = []
        question_index = 0
        for context in data:
            links = {l['target'].lower(): l['target'] for l in context['links']}
            for q in context['questions']:
                example = {'qid': q['qid'] if 'qid' in q else str(question_index),
                           'phrase': q['question'],
                           'context': self.preprorcess_context_method(q['context'], q['question'], context['title'] + ': ' + context['text'], q, question_index, links),
                           'answer': self.preprocess_iirc_answer(q['answer']),
                           'answer_type': q['answer']['type']}
                examples.append(example)
                question_index += 1

        self.data = pd.DataFrame([[example['qid'],
                                   example['phrase'],
                                   example['context'],
                                   example['answer'],
                                   example['answer_type']]
                                  for example in examples], columns=['qid', 'phrase', 'context', 'answer', 'type'])

        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self.qids = self.data.qid
        self.phrases = self.data.phrase
        self.contexts = self.data.context
        self.gold = self.data.answer
        self.types = self.data.type

        self.tokenizer = tokenizer
        self.source_len = max_seq_len
        self.summ_len = summ_len

    def preprocess_retrieved_sentece(self, sentence):
        #return sentence
        if sentence[:12] == 'Introduction':
            # remove the prefix
            return '\n\n'.join(sentence.split('\n\n')[1:])
        return sentence

    def preprocess_iirc_context_train(self, context, question, main_passage, q, i, links):
        """
        preprocess a context from iirc dataset to a text, which can be used by a generative model
        """
        gold_sentences = [c['passage'] + ': ' + c['text'] for c in context
                          if c['passage'] != 'main']
        return 'Links: \n' + '\n'.join(gold_sentences) + '\n Main: \n' + main_passage

    def preprocess_iirc_context_with_retrieval(self, context, question, main_passage, q, i, links):
        """
        preprocess a context from iirc dataset to a text, which can be used by a generative model
        """
        retrieved_sentences = [links[r['passage']] + ': ' + r['text'].replace('\n', ' ')  for r in self.retrieved_contexts[i]
                               if 'NULL' not in r['text']]
        return 'Links: \n' + '\n'.join(retrieved_sentences) + '\n Main: \n' + main_passage

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
        source_text = 'QA: ' + source_text

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
