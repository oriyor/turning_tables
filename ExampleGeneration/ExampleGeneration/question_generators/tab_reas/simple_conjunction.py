import hashlib
import random
import json, re
import logging
from copy import copy

from ExampleGeneration.common.table_wrapper import WikiTable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sample = []
question_index = {'i': 0}
from ExampleGeneration.question_generators.question_generator import QuestionGenerator
from ExampleGeneration.common.multiqa_format_wrapper import Question, SyntheticQuestion

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SimpleConjunction(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'SimpleConjunction'
        self.reasoning_types = ['conjunction']
        super().__init__(args)
        self._qgen_config = qgen_config

    def filter_compoisition_distractor(self, f, f1, f2):
        """
        check if f can be a distractor for a compoisiton question from f1 to f2
        """
        # filter if this distractor is the explicit answer to the question
        if f.target_column_ind == f2.target_column_ind and f.src_column_ind == f1.src_column_ind:
            if f.source_val_indices[0] == f1.source_val_indices[0]:
                return True

        # filter if the distractor columns are irrelevent
        relevant_columns = {f1.src_column_ind, f2.src_column_ind, f2.target_column_ind}
        if len({f.src_column_ind, f.target_column_ind}.intersection(relevant_columns)) < 2:
            return True

        # else return false
        return False

    def generate(self, context):
        """
        :param from_question:
        :param to_question:
        :return: the composition question returned by injection of the first question to the second
        """
        # Generate facts
        table = WikiTable(context)
        all_facts = self.generate_facts(table)
        facts = [f for f in all_facts
                 if not f.filtered]

        # init fields
        random.seed(42)
        key_columns = table.get_key_column()
        conjunction_questions = []

        for temlates_config in self._qgen_config['templates']:
            template = temlates_config['question_template']

            # generate composition questions by looping the facts
            for f1 in facts:

                f1_reversed = self.reverse_fact(f1, all_facts)
                if f1_reversed is not None:
                    # look for facts about the the key columns
                    if f1.target_column_ind in key_columns:
                        source_column = f1.src_column_ind
                        target_column = f1.target_column_ind

                        for f2 in facts:

                            f2_reversed = self.reverse_fact(f2, all_facts)
                            # look for fact pairs with multiple answer, and more than one shared answer
                            if (f2.target_column_ind == target_column) \
                                    and len(f1.source_val_indices) > 1 and len(f2.source_val_indices) > 1 \
                                    and set(f1.source_val_indices).intersection(set(f2.source_val_indices)) \
                                    and f2_reversed is not None:

                                # the answer is the intersection between the facts
                                # it needs to be not empty, and not equal to either of the original sets
                                answer = list(set(f1.target_column_values) \
                                              .intersection(set(f2.target_column_values)))

                                if len(answer) and len(answer) != len(set(f1.target_column_values)) \
                                        and len(answer) != len(set(f2.target_column_values)):
                                    phrase = template
                                    phrase = phrase.replace("[page_title]", f1.page_title)
                                    phrase = phrase.replace("[table_title]", f1.table_title.strip())
                                    phrase = phrase.replace("[val1]", str(f1.src_column_value).strip())
                                    phrase = phrase.replace("[val2]", str(f2.src_column_value).strip())
                                    phrase = phrase.replace("[source_column1]", f1.formatted_src_column_header.strip())
                                    phrase = phrase.replace("[source_column2]", f2.formatted_src_column_header.strip())

                                    # we need to change to plural depending on the answer
                                    target_header = f2.formatted_target_column_header.strip()
                                    if len(answer) > 1:
                                        phrase = phrase.replace("[target_column]", f'{target_header}s')
                                        phrase = phrase.replace("[WH]", 'where')

                                    else:
                                        phrase = phrase.replace("[target_column]", f'{target_header}')
                                        phrase = phrase.replace("[WH]", 'was')

                                    # we can use the same sampling as sample composition here
                                    possible_distractors = self.sample_distractors(facts, f1, f2, 'composition')
                                    distractors = random.sample(possible_distractors, min(len(possible_distractors), 6))

                                    m = hashlib.md5()
                                    m.update(context.id.encode())
                                    m.update(str(source_column).encode())
                                    m.update(str(target_column).encode())
                                    m.update(f1.src_column_value.encode())
                                    m.update(f2.src_column_value.encode())

                                    qid = 'Conjunction-' + m.hexdigest()

                                    # reverse facts with equal probability
                                    # update the template and facts accordingly
                                    formatted_facts = []
                                    question_template = 'conjunction'
                                    for question_fact in [
                                        {'fact': f1, 'reversed_fact': f1_reversed, 'template_suffix': 'first_reversed'},
                                        {'fact': f2, 'reversed_fact': f2_reversed,
                                         'template_suffix': 'second_reversed'}]:

                                        reverse_fact = random.choice([True, False])

                                        if reverse_fact:
                                            # if reversing the fact, add the suffix to the template
                                            formatted_facts.extend(question_fact['reversed_fact'])
                                            template_suffix = question_fact['template_suffix']
                                            question_template = f'{question_template}-{template_suffix}'

                                        else:
                                            # else, use the regular fact
                                            formatted_facts.append(question_fact['fact'])

                                    conjunction_questions.append(SyntheticQuestion(qid=qid,
                                                                                   question=phrase,
                                                                                   answers=answer,
                                                                                   facts=[f.format_fact() for f in
                                                                                          formatted_facts],
                                                                                   distractors=[d.format_fact() for d in
                                                                                                distractors],
                                                                                   metadata={'type': 'conjunction',
                                                                                             'reasoning': self.reasoning_types,
                                                                                             'answer_type': 'entity',
                                                                                             'reversed_facts': [
                                                                                                                   f.format_fact()
                                                                                                                   for f
                                                                                                                   in
                                                                                                                   f2_reversed]
                                                                                                               + [
                                                                                                                   f.format_fact()
                                                                                                                   for f
                                                                                                                   in
                                                                                                                   f1_reversed],
                                                                                             'template': question_template
                                                                                             },
                                                                                   )
                                                                 )

        if len(conjunction_questions) > 10:
            conjunction_questions = random.sample(conjunction_questions, 10)

        return conjunction_questions
