import hashlib
import random
import json, re
import logging
from copy import copy
import numpy as np
from ExampleGeneration.common.table_wrapper import WikiTable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sample = []
question_index = {'i': 0}
from ExampleGeneration.question_generators.question_generator import QuestionGenerator
from ExampleGeneration.common.multiqa_format_wrapper import Question, SyntheticQuestion

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class OnlyQuantifier(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'only_quantifier'
        self.reasoning_types = ['quantifiers']
        super().__init__(args)
        self._qgen_config = qgen_config
        self.operator = 'the only'

    def every_quantifier(self, value, target_values_list):
        """
        check whether the value is in everyone of the lists
        """
        appearance_vector = [True
                             if value in values_list else False
                             for values_list in target_values_list]

        # check if the value appears in more than half of the lists
        if np.average(appearance_vector) == 1:
            return True
        return False

    def most_quantifier(self, value, target_values_list):
        """
        check whether the value is in most of the lists
        """
        appearance_vector = [True
                             if value in values_list else False
                             for values_list in target_values_list]

        # check if the value appears in more than half of the lists
        if np.average(appearance_vector) > 0.5:
            return True
        return False

    def any_quantifier(self, value, target_values_list):
        """
        check whether the value is in any of the lists
        """
        for values_list in target_values_list:
            if value in values_list:
                return True
        return False

    def only_quantifier(self, value, target_values_list):
        """
        check whether the value appears only in the list we are examining
        """
        # first check in how many lists the value appears
        value_appearances_vector = [True if value in target_list else False
                                    for target_list in target_values_list]

        # if the value appears in more then one list, return False, else True
        return np.sum(value_appearances_vector) <= 1

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

        # filter facts to facts about arithmetic columns
        key_columns = table.get_key_column()
        facts = [f for f in facts
                 if f.src_column_ind in key_columns]

        # init fiels
        random.seed(42)
        quantifier_questions = []


        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']



        # generate superlative questions by looping the columns

            for target_column in range(table.m):
                for source_column in key_columns:

                    # get all the facts between the columns
                    relevant_facts = [f for f in facts
                                      if f.target_column_ind == target_column
                                      and f.src_column_ind == source_column]

                    # verify the source indices are equal to the entire column
                    source_indices = {index for f in relevant_facts
                                      for index in f.source_val_indices}

                    if len(source_indices) == table.m:

                        # we need this in order to calculate the answer
                        target_vals_lists = [f.target_column_values for f in relevant_facts]

                        for f in relevant_facts:

                            # we need to look at the source value and target values for every fact
                            source_val = f.src_column_value

                            for target_val in f.target_column_values:
                                # calculate phrase and qid
                                phrase = template
                                phrase = phrase.replace("[page_title]", relevant_facts[0].page_title)
                                phrase = phrase.replace("[table_title]", relevant_facts[0].table_title.strip())
                                phrase = phrase.replace("[source_column]", relevant_facts[0].src_column_header.strip())
                                phrase = phrase.replace("[target_column]", relevant_facts[0].target_column_header.strip())
                                phrase = phrase.replace("[to_cell_text]", target_val.strip())
                                phrase = phrase.replace("[from_cell_text]", source_val.strip())

                                phrase = phrase.replace("[QUANTIFIER_OPERATOR]", self.operator)

                                m = hashlib.md5()
                                m.update(context.id.encode())
                                m.update(str(source_column).encode())
                                m.update(str(target_column).encode())
                                m.update(target_val.encode())
                                m.update(source_val.encode())

                                # the answer here will be the quantifier operation
                                bool_answer = self.only_quantifier(value=target_val, target_values_list=target_vals_lists)

                                qid = 'Only-' + m.hexdigest()

                                str_answer = 'yes' if bool_answer else 'no'

                                quantifier_questions.append(SyntheticQuestion(qid=qid,
                                                                              question=phrase,
                                                                              answers=[str_answer],
                                                                              facts=[f.format_fact()
                                                                                     for f in relevant_facts],
                                                                              distractors=[],
                                                                              metadata={'type': f'only_quantifier',
                                                                                        'reasoning': self.reasoning_types,
                                                                                        'answer_type': 'boolean',
                                                                                        "reversed_facts": "",
                                                                                        "template": "only_quantifier"
                                                                                        }
                                                                              )
                                                            )

        # sample
        quantifier_questions_true = [q for q in quantifier_questions
                                     if 'yes' in q.answers]
        quantifier_questions_false = [q for q in quantifier_questions
                                      if 'no' in q.answers]

        # num questions to sample
        # sample at most 4 questions for each type
        num_questions_for_answer = min(4, min(len(quantifier_questions_true), len(quantifier_questions_false)))
        quantifier_questions_false = random.sample(quantifier_questions_false, num_questions_for_answer)
        quantifier_questions_true = random.sample(quantifier_questions_true, num_questions_for_answer)

        return quantifier_questions_true + quantifier_questions_false

