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


class Counting(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'Counting'
        self.reasoning_types = ['Counting']
        super().__init__(args)
        self._qgen_config = qgen_config

    def filter_counting_distractor(self, f, f1):
        """
        check if f can be a distractor for a counting question about f
        """

        # if the facts are about different indices in the relevant columns, keep
        relevant_columns = {f1.src_column_ind, f1.target_column_ind}
        if len({f.src_column_ind, f.target_column_ind}.intersection(relevant_columns)) == 2:
            if not set(f.source_val_indices).intersection(set(f1.source_val_indices)):
                return False

        # else filter
        return True

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
        counting_questions = []
        key_columns_all = table.get_key_column()
        entities_columns = table.get_entities_columns()
        key_columns = key_columns_all.union(entities_columns)

        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']

            # generate counting questions by looping the facts
            for f in facts:

                f_reversed = self.reverse_fact(f, all_facts)

                # look for facts about the the key columns
                if f.target_column_ind in key_columns:
                    source_column = f.src_column_ind
                    target_column = f.target_column_ind

                    # the answer is the count of the group
                    answer = len(list(set(f.target_column_values)))

                    if f_reversed is not None and answer > 1:
                        phrase = template
                        phrase = phrase.replace("[page_title]", f.page_title)
                        phrase = phrase.replace("[table_title]", f.table_title.strip())
                        phrase = phrase.replace("[val_1]", str(f.src_column_value).strip())
                        phrase = phrase.replace("[source_column]", f.formatted_src_column_header.strip())

                        # we need to account for plural
                        formatted_target_header = f.formatted_target_column_header.strip()
                        if formatted_target_header[-1] == 's':
                            phrase = phrase.replace("[target_column]", formatted_target_header)
                        elif formatted_target_header[-2:] == 's)':
                            phrase = phrase.replace("[target_column]", formatted_target_header.replace('(s)', 's'))
                        else:
                            phrase = phrase.replace("[target_column]", f'{formatted_target_header}s')

                        # sample distractors
                        possible_distractors = self.sample_distractors(facts, f, f, 'counting')
                        num_distractors = min(len(possible_distractors), 4)
                        distractors = random.sample(possible_distractors, num_distractors)

                        m = hashlib.md5()
                        m.update(context.id.encode())
                        m.update(str(source_column).encode())
                        m.update(str(target_column).encode())
                        m.update(f.src_column_value.encode())

                        qid = 'Counting-' + m.hexdigest()

                        # we will use reverse facts with equal probability
                        reverse_facts = random.choice([True, False])
                        if reverse_facts:
                            question_facts = [f.format_fact() for f in f_reversed]
                            question_template = 'counting_reversed'

                        else:
                            question_facts = [f.format_fact()]
                            question_template = 'counting_list'

                        counting_questions.append(SyntheticQuestion(qid=qid,
                                                                    question=phrase,
                                                                    answers=[answer],
                                                                    facts=question_facts,
                                                                    distractors=[d.format_fact() for d in
                                                                                 distractors],
                                                                    metadata={'type': 'counting',
                                                                              'reasoning': self.reasoning_types,
                                                                              'answer_type': 'numeric',
                                                                              'reversed_facts': [f.format_fact() for f in
                                                                                                 f_reversed],
                                                                              'source_column': f.formatted_target_column_header.strip(),
                                                                              'template': question_template
                                                                              }
                                                                    )
                                                  )

        if len(counting_questions) > 4:
            counting_questions = random.sample(counting_questions, 4)

        return counting_questions
