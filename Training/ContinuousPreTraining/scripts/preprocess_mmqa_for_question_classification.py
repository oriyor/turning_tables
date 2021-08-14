import json
import gzip
from datasets import tqdm


def main():

    def parse_image_question_classifier(q):
        """
        parse a question to tell if it's an image question
        """
        question = q['question']
        id = q['qid']

        # if this is a test question, we don't no the answer
        if ds_split['test']:
            ds_array.append({'context': question,
                             'answer': "",
                             'all_answers': [""],
                             'id': id})
        else:
            # check if the question has an image modality
            question_modalities = q['metadata']['modalities']
            if 'image' in question_modalities:
                answer = 'yes'
            else:
                answer = 'no'

            ds_array.append({'context': question,
                             'answer': answer,
                             'all_answers': [answer],
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

    train = {'lines': train, 'array': parsed_train_questions, 'balanced_sampling': True, 'test': False,
             'add_table': False}
    dev = {'lines': dev, 'array': parsed_dev_questions, 'balanced_sampling': False, 'test': False, 'add_table': False}
    test = {'lines': test, 'array': parsed_test_questions, 'balanced_sampling': False, 'test': True, 'add_table': False}

    print(f'Preprocessing MMQA for image question classification')
    for ds_split in [train, dev, test]:

        ds_lines = ds_split['lines']
        ds_array = ds_split['array']

        for q in tqdm(ds_lines):

            parse_image_question_classifier(q)

    # write train questions
    train_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_question_classifier_train.json'
    output_fp = gzip.open(train_output_file, 'wb')
    for question in parsed_train_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    # write dev questions
    dev_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_question_classifier_dev.json'
    output_fp = gzip.open(dev_output_file, 'wb')
    for question in parsed_dev_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    # write test questions
    dev_output_file = 'ContinuousPreTraining/Data/mmqa/parsed_mmqa_question_classifier_test.json'
    output_fp = gzip.open(dev_output_file, 'wb')
    for question in parsed_test_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    print('Finished pre-processing MMQA for question classification')


if __name__ == '__main__':
    """
    Script to preprocess MMQA for image question classification
    """
    main()
