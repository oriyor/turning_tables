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


class Quantifiers(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'quantifiers'
        self.reasoning_types = ['quantifiers']
        super().__init__(args)
        self._qgen_config = qgen_config
        self.ops = {'most':
            {
                'template_string': [{'text': 'do most', 'plural': True}],
                'func': self.most_quantifier
            },
            'every': {
                'template_string': [{'text': 'does every', 'plural': False}],
                # ,{'text': 'do all', 'plural': True}],
                'func': self.every_quantifier
            }
        }

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

        # init fields
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

                        target_vals = {val for f in relevant_facts
                                       for val in f.target_column_values}
                        for val in target_vals:

                            for op in self.ops:
                                # calculate phrase and qid
                                phrase = template
                                phrase = phrase.replace("[page_title]", relevant_facts[0].page_title)
                                phrase = phrase.replace("[table_title]", relevant_facts[0].table_title.strip())

                                # check for plural source column
                                template_string = random.choice(self.ops[op]['template_string'])
                                source_column_text = relevant_facts[0].src_column_header.strip()
                                if template_string['plural']:
                                    source_column_text += '(s)'

                                phrase = phrase.replace("[source_column]", source_column_text)
                                phrase = phrase.replace("[target_column]",
                                                        relevant_facts[0].target_column_header.strip())
                                phrase = phrase.replace("[to_cell_text]", val.strip())
                                phrase = phrase.replace("[QUANTIFIER_OPERATOR]", template_string['text'])

                                m = hashlib.md5()
                                m.update(context.id.encode())
                                m.update(str(source_column).encode())
                                m.update(str(target_column).encode())
                                m.update(op.encode())

                                # the answer here will be the quantifier operation
                                bool_answer = self.ops[op]['func'](value=val, target_values_list=[f.target_column_values
                                                                                                  for f in
                                                                                                  relevant_facts])

                                qid = 'Quantifier-' + m.hexdigest()
                                str_answer = 'yes' if bool_answer else 'no'
                                template_text = template_string['text']
                                quantifier_questions.append(SyntheticQuestion(qid=qid,
                                                                              question=phrase.capitalize(),
                                                                              answers=[str_answer],
                                                                              facts=[f.format_fact()
                                                                                     for f in relevant_facts],
                                                                              distractors=[],
                                                                              metadata={'type': f'{op}_quantifier',
                                                                                        'reasoning': self.reasoning_types,
                                                                                        'answer_type': 'boolean',
                                                                                        "reversed_facts": "",
                                                                                        "template": f'{op}_{template_text}_quantifier'
                                                                                        }
                                                                              )
                                                            )


        # we want to sample equally between the different types
        # traverse the different quetion types
        sampled_questions = []
        for gold_op in self.ops:
            questions_for_type = [q for q in quantifier_questions
                                  if q.metadata['type'] == f'{gold_op}_quantifier']

            # sample
            type_questions_true = [q for q in questions_for_type
                                   if 'yes' in q.answers]
            type_questions_false = [q for q in questions_for_type
                                    if 'no' in q.answers]

            # sample at most 4 questions for each type
            num_questions_for_answer = min(4, min(len(type_questions_true), len(type_questions_false)))
            type_questions_true = random.sample(type_questions_true, num_questions_for_answer)
            type_questions_false = random.sample(type_questions_false, num_questions_for_answer)

            sampled_questions += type_questions_true
            sampled_questions += type_questions_false

        return sampled_questions
