import json
import gzip


def main():
    # read files
    with open('ContinuousPreTraining/Data/drop/drop_dataset_dev.json') as json_file:
        dev = json.load(json_file)

    with open('ContinuousPreTraining/Data/drop/drop_dataset_train.json') as json_file:
        train = json.load(json_file)

    def parse_drop_answer(answer, train_mode=True):
        """
        parse drop answer
        """
        if len(answer['spans']) == 1:
            # print('span')
            return ' '.join(answer['spans'])

        if len(answer['spans']) > 1:
            if train_mode:
                # print('spans')

                return '#'.join(answer['spans'])
            else:
                return answer['spans']

        elif len(answer['number']) > 0:
            # print('number')
            return answer['number']

        else:
            # print('date')
            day = answer['date']['day']
            if day:
                day += ' '

            month = answer['date']['month']
            if month:
                month += ' '

            year = answer['date']['year']

            return f'{day}{month}{year}'

    # go over all ds splits
    parsed_train_questions = []
    parsed_dev_questions = []

    for context in train.values():

        context_passage = context['passage']

        for q in context['qa_pairs']:
            context_with_question = q['question'] + '\n' + context_passage
            parsed_train_questions.append({'context': context_with_question,
                                           'answer': parse_drop_answer(q['answer']),
                                           'all_answers': parse_drop_answer(q['answer']),
                                           'id': q['query_id']})

    for context in dev.values():

        context_passage = context['passage']

        for q in context['qa_pairs']:
            context_with_question = q['question'] + '\n' + context_passage
            answer = parse_drop_answer(q['answer'])
            validated_answers = [parse_drop_answer(annotated_answer, train_mode=False)
                                 for annotated_answer in q['validated_answers']] + [
                                    parse_drop_answer(q['answer'], train_mode=False)]
            parsed_dev_questions.append({'context': context_with_question,
                                         'answer': answer,
                                         'all_answers': validated_answers,
                                         'id': q['query_id']})

    # write train questions
    train_output_file = 'ContinuousPreTraining/Data/drop/parsed_drop_train_with_lists.json'
    output_fp = gzip.open(train_output_file, 'wb')
    for question in parsed_train_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    # write dev questions
    dev_output_file = 'ContinuousPreTraining/Data/drop/parsed_drop_dev_with_lists.json'
    output_fp = gzip.open(dev_output_file, 'wb')
    for question in parsed_dev_questions:
        output_fp.write((json.dumps(question, ensure_ascii=False) + '\n').encode('utf-8'))

    print('Finished pre-processing drop')

if __name__ == '__main__':
    """
    Script to preprocess the drop dataset to UnifiedQA format
    """
    main()
