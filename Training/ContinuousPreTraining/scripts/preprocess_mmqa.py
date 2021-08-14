import json
import gzip
from datasets import tqdm
from transformers import T5Tokenizer
import matplotlib.pyplot as plt


def main():
    def parse_question(q, linearized_table):
        """
        parse the question to conain the oracle text documentss and table
        """
        question = q['question']
        id = q['qid']
        context_text_docs = q['metadata']['text_doc_ids']
        question_text_docs = [texts_dict[doc] for doc in context_text_docs]

        supporting_contexts_ids = set([supporting_context['doc_id']
                                       for supporting_context in q['supporting_context']
                                       if supporting_context['doc_part'] == 'text'])

        gold_text_contexts = [d for d in question_text_docs
                              if d['id'] in supporting_contexts_ids]
        text_context = '\n'.join([d['text'] for d in gold_text_contexts])

        all_answers = [str(a['answer']) for a in q['answers']]
        answer = '#'.join(all_answers)

        ds_array.append({'context': f'{question} \n {text_context} \n {linearized_table}',
                         'answer': answer,
                         'all_answers': [all_answers],
                         'id': id})


    train = []
    dev = []
    test = []
    tables = []
    texts = []

    f = gzip.open("ContinuousPreTraining/Data/mmqa/MMQA_train.jsonl.gz?raw=true", 'r')
    for l in f:
        train.append(json.loads(l))

    f = gzip.open("ContinuousPreTraining/Data/mmqa/MMQA_dev.jsonl.gz?raw=true", 'r')
    for l in f:
        dev.append(json.loads(l))

    f = gzip.open("ContinuousPreTraining/Data/mmqa/MMQA_test.jsonl.gz?raw=true", 'r')
    for l in f:
        test.append(json.loads(l))

    f = gzip.open("ContinuousPreTraining/Data/mmqa/MMQA_tables.jsonl.gz?raw=true", 'r')
    for l in f:
        tables.append(json.loads(l))

    f = gzip.open("ContinuousPreTraining/Data/mmqa/MMQA_texts.jsonl.gz?raw=true", 'r')
    for l in f:
        texts.append(json.loads(l))

    # go over all ds splits
    parsed_train_questions = []
    parsed_dev_questions = []
    parsed_test_questions = []
    tables_dict = {t['id']: t for t in tables}
    texts_dict = {t['id']: t for t in texts}

    train = {'lines': train, 'array': parsed_train_questions, 'balanced_sampling': True, 'test': False,
             'add_table': False}
    dev = {'lines': dev, 'array': parsed_dev_questions, 'balanced_sampling': False, 'test': False, 'add_table': False}

    print(f'Preprocessing MMQA')
    for ds_split in [train, dev]:

        ds_lines = ds_split['lines']
        ds_array = ds_split['array']

        for q in tqdm(ds_lines):

            # filter image questions
            if 'image' not in q['metadata']['modalities']:

                # liniearize table
                linearized_table = ''
                table_id = q['metadata']['table_id']
                table = tables_dict[table_id]['table']
                for i, row in enumerate(table['table_rows']):

                    row_text = f'R{i}: '
                    for j, cell in enumerate(row):
                        column_name = table['header'][j]['column_name']
                        column_value = cell['text']
                        row_text += f'{column_name} is {column_value};'
                    linearized_table += row_text

                parse_question(q, linearized_table)
            # parse_question_paragraph_classifier(q)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenized_inputs = [tokenizer.tokenize(input['context']) for input in dev['array']]
    input_lenghts = [len(input) for input in tokenized_inputs]
    plt.hist(input_lenghts, bins=128, cumulative=True, density=True)
    plt.plot()
    plt.title('Cumulative Linearized Tables Lengths')
    plt.xlabel('# tokens')
    plt.xlabel('% of example')
    plt.grid()
    plt.show()
    # write train questions
    train_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_train_oracle.json'
    output_fp = gzip.open(train_output_file, 'wb')
    for question in parsed_train_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    # write dev questions
    dev_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_dev_oracle.json'
    output_fp = gzip.open(dev_output_file, 'wb')
    for question in parsed_dev_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    print('Finished pre-processing MMQA')


if __name__ == '__main__':
    """
    Script to preprocess the MMQA dataset to UnifiedQA format
    """
    main()
