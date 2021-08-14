import random

from ExampleGeneration.common.multiqa_format_wrapper import Question, Answer

def sample_questions_per_template(questions, sample_size):
    """
    :param questions:
    :param sample_size:
    :return: for each question template sample take sample_size
    note we will restart the random seed for each template, so to enforce the order to be
    reproducible.
    """

    template_list = [q.metadata['template_variation'] for q in questions]
    sampled_questions = []
    for template_name in set(template_list):
        curr_template_questions = [questions[i] for i,t in enumerate(template_list) if t == template_name]
        if len(curr_template_questions) < sample_size:
            sampled_questions += curr_template_questions
        else:
            # we will restart the random seed for each template, so to enforce the order to be
            # reproducible. The shuffle insures that if we decide to sample more from each context the first
            # K we already sampled remain the same
            random.seed(7)
            random.shuffle(curr_template_questions)
            sampled_questions += curr_template_questions[0:sample_size]

    return sampled_questions

def sample_questions(questions, sample_size):
    """
    :param questions:
    :param sample_size:
    :return: a random sample_size of items from questions
    """
    random.seed(3)
    if len(questions) < sample_size:
        return questions
    return random.sample(questions, sample_size)

def get_composite_question(first_comp_question, second_comp_question):
    """
    :param first_comp_question:
    :param second_comp_question:
    :return: the composition question returned by injection of the first question to the second
    """
    first_comp_answer = first_comp_question.answers.answers[0]
    question_text = second_comp_question.question.replace(first_comp_answer,
                                                          f'({first_comp_question.question} : {first_comp_answer})')
    return Question(qid=f'Comp-{first_comp_question.qid}-{second_comp_question.qid}',
                    question=question_text,
                    answers=second_comp_question.answers,
                    metadata={
                        'type': f'composition-{first_comp_question.metadata["type"]}-{second_comp_question.metadata["type"]}',
                        'source': 'generated-composition', "schema": "simple-injection-composition",
                        'link_answer': first_comp_answer},
                    supporting_context=first_comp_question.supporting_context + second_comp_question.supporting_context)


def get_conjunction_question(q1, q2, intersecting_answers):
    """
    :param q1:
    :param q2:
    :param intersecting_answers:
    :return: the conjunction question returned by combining the two questions
    """
    if random.choice([True, False]):
        question_text = f'({q1.question}) and ({q2.question})'
    else:
        question_text = f'({q2.question}) and ({q1.question})'
    qid = f'Conj-{q1.qid}-{q2.qid}'
    return Question(qid=qid,
                    question=question_text,
                    answers=Answer(list(intersecting_answers)),
                    metadata={
                        'type': f'conj-{q1.metadata["type"]}-{q2.metadata["type"]}',
                        'source': 'generated-conjunction', "schema": "conj1",
                        'q1_answers': q1.answers.answers,
                        'q2_answers': q2.answers.answers},
                    supporting_context=q1.supporting_context + q2.supporting_context)
