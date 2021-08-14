import argparse
import json
import gzip
from datasets import tqdm
from transformers import T5Tokenizer
import matplotlib.pyplot as plt
import pandas as pd


def main(dev_questions_classifier_predictions_path,
         test_questions_classifier_predictions_path,
         dev_paragraphs_classifier_predictions_path,
         test_paragraphs_classifier_predictions_path
         ):

    # read files
    dev = []
    test = []
    tables = []
    texts = []

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

    tables_dict = {t['id']: t for t in tables}
    texts_dict = {t['id']: t for t in texts}

    # read classified questions
    classified_questions_dev_df = pd.read_csv(dev_questions_classifier_predictions_path)
    classified_questions_dev = {q['id'] for q in classified_questions_dev_df.transpose().to_dict().values() if
                                'no' in q['prediction']}

    classified_questions_test_df = pd.read_csv(test_questions_classifier_predictions_path)
    classified_questions_test = {q['id'] for q in classified_questions_test_df.transpose().to_dict().values() if
                                 'no' in q['prediction']}
    classified_questions = classified_questions_test.union(classified_questions_dev)

    # read classified text docs
    classified_pargraphs_dev_df = pd.read_csv(dev_paragraphs_classifier_predictions_path)
    classified_pargraphs_test_df = pd.read_csv(test_paragraphs_classifier_predictions_path)

    # concat relevant text docs
    question_to_classified_text_docs = {}
    classified_text_docs_dev = [v for v in classified_pargraphs_dev_df.transpose().to_dict().values()]
    classified_text_docs_test = [v for v in classified_pargraphs_test_df.transpose().to_dict().values()][:36600]

    for classified_text_doc in classified_text_docs_dev + classified_text_docs_test:
        if classified_text_doc['id'] not in question_to_classified_text_docs:
            question_to_classified_text_docs[classified_text_doc['id']] = ''
        if 'yes' in classified_text_doc['prediction']:
            question_to_classified_text_docs[classified_text_doc['id']] += '\n'
            curr_text = '\n'.join(classified_text_doc['question'].split('\n')[1:])
            question_to_classified_text_docs[classified_text_doc['id']] += curr_text

    # go over all ds splits
    parsed_dev_questions = []
    parsed_test_questions = []

    dev = {'lines': dev, 'array': parsed_dev_questions, 'train': False, 'test': False}
    test = {'lines': test, 'array': parsed_test_questions, 'train': False, 'test': True}

    cnt = 0
    for ds_split in [dev, test]:
        ds_lines = ds_split['lines']
        ds_array = ds_split['array']

        for q in tqdm(ds_lines):

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

            # we want all questions that are either classified or table / text questions for train
            if q['qid'] in classified_questions or (ds_split['train'] and 'image' not in q['metadata']['modalities']):
                cnt += 1

                question = q['question']
                id = q['qid']

                # get doc text
                doc_text = ""
                if ds_split['train']:
                    supporting_contexts_ids = set([supporting_context['doc_id']
                                                   for supporting_context in q['supporting_context']
                                                   if supporting_context['doc_part'] == 'text'])
                    for supporting_contexts_id in supporting_contexts_ids:
                        doc_text += '\n'
                        doc_text += texts_dict[supporting_contexts_id]['text']

                else:
                    doc_text += question_to_classified_text_docs[id]

                # for test we don't know the answers
                if ds_split['test']:
                    ds_array.append({'context': f'{question} {doc_text} \n {linearized_table}',
                                     'answer': "",
                                     'all_answers': [],
                                     'id': id})
                else:
                    all_answers = [str(a['answer']) for a in q['answers']]
                    answer = '#'.join(all_answers)
                    ds_array.append({'context': f'{question} {doc_text} \n {linearized_table}',
                                     'answer': answer,
                                     'all_answers': [all_answers],
                                     'id': id})

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenized_inputs = [tokenizer.tokenize(input['context']) for input in tqdm(dev['array'])]
    input_lenghts = [len(input) for input in tokenized_inputs]
    plt.hist(input_lenghts, bins=128, cumulative=True, density=True)
    plt.plot()
    plt.title('Cumulative Linearized Tables Lengths')
    plt.xlabel('# tokens')
    plt.xlabel('% of example')
    plt.grid()
    plt.show()

    # write dev questions
    dev_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_dev_retrieval.json'
    output_fp = gzip.open(dev_output_file, 'wb')
    for question in parsed_dev_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    # write test questions
    test_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_test_retrieval.json'
    output_fp = gzip.open(test_output_file, 'wb')
    for question in parsed_test_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    print('Finished creating MMQA retrieval contexts')


if __name__ == "__main__":
    """
    Class to create unify MMQA contexts after running question and paragraph classification
    """
    parser = argparse.ArgumentParser(description='Unify MMQA contexts')
    parser.add_argument("--dev_questions_classifier_predictions_path",
                        type=str,
                        required=False,
                        help='location of the dev classifier predictions file')
    parser.add_argument("--test_questions_classifier_predictions_path",
                        type=str,
                        required=False,
                        help='location of the test classifier predictions file')
    parser.add_argument("--dev_paragraphs_classifier_predictions_path",
                        type=str,
                        required=False,
                        help='location of the dev classifier predictions file')
    parser.add_argument("--test_paragraphs_classifier_predictions_path",
                        type=str,
                        required=False,
                        help='location of the test classifier predictions file')

    args = parser.parse_args()
    main(args.dev_questions_classifier_predictions_path,
         args.test_questions_classifier_predictions_path,
         args.dev_paragraphs_classifier_predictions_path,
         args.test_paragraphs_classifier_predictions_path
         )
