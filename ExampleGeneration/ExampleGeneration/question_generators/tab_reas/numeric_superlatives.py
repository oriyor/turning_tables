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


class NumericSuperlatives(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'superlatives'
        super().__init__(args)
        self.temporal = qgen_config['temporal']
        self.reasoning_types = ['temporal_superlatives'] if self.temporal else ['numeric_superlatives']
        self._qgen_config = qgen_config

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

        arithmetic_columns = table.get_arithmetic_inds()

        if self.temporal:
            numeric_columns = [col for col in arithmetic_columns
                               if self.is_date_column(table, col)]

        else:
            numeric_columns = [col for col in arithmetic_columns
                               if not self.filter_date_columns(table, col)]

        # filter facts to facts about arithmetic columns
        facts = [f for f in facts
                 if f.target_column_ind in numeric_columns]

        # we need to split between singular and plural for numeric
        if self.temporal:
            higher_comparator = self._qgen_config['templates'][0]['numeric_higher_comparator']
            lower_comparator = self._qgen_config['templates'][0]['numeric_lower_comparator']
        else:
            higher_comparator_singular = self._qgen_config['templates'][0]['numeric_higher_comparator_singular']
            lower_comparator_singular = self._qgen_config['templates'][0]['numeric_lower_comparator_singular']
            higher_comparator_plural = self._qgen_config['templates'][0]['numeric_higher_comparator_plural']
            lower_comparator_plural = self._qgen_config['templates'][0]['numeric_lower_comparator_plural']

        random.seed(42)

        # generate superlative questions by looping the columns
        superlative_questions = []
        target_numeric_columns = numeric_columns
        key_columns_all = table.get_key_column()
        entities_columns = table.get_entities_columns()
        key_columns = key_columns_all.union(entities_columns)

        source_key_columns = [k for k in key_columns
                              if k not in target_numeric_columns]

        for config_template in self._qgen_config['templates']:
            template = config_template['question_template']

            for target_column in target_numeric_columns:
                for source_column in source_key_columns:

                    # get all the facts between the columns
                    relevant_facts = [f for f in facts
                                      if f.target_column_ind == target_column
                                      and f.src_column_ind == source_column]

                    # verify the source indices are equal to the entire column
                    source_indices = {index for f in relevant_facts
                                      for index in f.source_val_indices}

                    if len(source_indices) == table.m:

                        # check if we can parse the column

                        filter = False
                        try:
                            target_vals = [self.parse_value(val, self.temporal) for f in relevant_facts
                                           for val in f.target_column_values]

                        except:
                            filter = True

                        if not filter:
                            for use_higher_comparator in [True, False]:

                                # calculate phrase and qid
                                phrase = template
                                target_column_header = relevant_facts[0].formatted_target_column_header.strip()

                                phrase = phrase.replace("[page_title]", relevant_facts[0].page_title)
                                phrase = phrase.replace("[table_title]", relevant_facts[0].table_title.strip())
                                phrase = phrase.replace("[source_column]",
                                                        relevant_facts[0].formatted_src_column_header.strip())
                                phrase = phrase.replace("[target_column]", target_column_header)

                                # we need to choose the comparators for each state
                                if self.temporal:
                                    if use_higher_comparator:
                                        comparator = higher_comparator
                                    else:
                                        comparator = lower_comparator

                                # for numeric we need to seperate between singular and plural
                                if not self.temporal:
                                    # check if the column is singular or plural
                                    # we use a heuristic approach: if the target column ends with an s, and does not of a number
                                    # of prefix, it is plural
                                    if len(target_column_header) and target_column_header[-1] != 's' \
                                            or target_column_header.startswith('#') \
                                            or target_column_header.startswith('number'):
                                        comparator = higher_comparator_singular if use_higher_comparator else lower_comparator_singular

                                    else:
                                        comparator = higher_comparator_plural if use_higher_comparator else lower_comparator_plural

                                phrase = phrase.replace("[COMPARATOR]", comparator)

                                m = hashlib.md5()
                                m.update(context.id.encode())
                                m.update(str(source_column).encode())
                                m.update(str(target_column).encode())
                                m.update(comparator.encode())
                                qid_prefix = 'Dt' if self.temporal else 'Nu'
                                qid = qid_prefix + m.hexdigest()

                                # we need to take the max/min based on the comparator
                                if use_higher_comparator:
                                    answer_value = max(target_vals)

                                else:
                                    answer_value = min(target_vals)

                                # the answer includes all the source values that have the correct target value
                                answer = [f.src_column_value for f in relevant_facts
                                          if answer_value in
                                          [self.parse_value(val, self.temporal)
                                           for val in f.target_column_values]]

                                if len(answer) == 1:

                                    # format facts
                                    if self.temporal:
                                        question_facts = [f.format_fact(date_time=random.choice([True, False]))
                                                          for f in relevant_facts]
                                    else:
                                        question_facts = [f.format_fact()
                                                          for f in relevant_facts]

                                    superlative_questions.append(SyntheticQuestion(qid=qid,
                                                                                   question=phrase,
                                                                                   answers=answer,
                                                                                   facts=question_facts,
                                                                                   distractors=[],
                                                                                   metadata={'type': ' '.join(
                                                                                       self.reasoning_types),
                                                                                       'reasoning': self.reasoning_types,
                                                                                       'answer_type': 'entity',
                                                                                       'reversed_facts': [],
                                                                                       'template': f'superlatives_{comparator}'
                                                                                   }
                                                                                   )
                                                                 )

        # sample
        if len(superlative_questions) > 4:
            superlative_questions = random.sample(superlative_questions, 4)

        return superlative_questions
