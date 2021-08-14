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


class SimpleComposition(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'SimpleComposition'
        self.reasoning_types = ['composition']
        super().__init__(args)
        self._qgen_config = qgen_config

    def filter_compoisition_distractor(self, f, f1, f2):
        """
        check if f can be a distractor for a compoisiton question from f1 to f2
        """

        # # filter if this distractor is the explicit answer to the question
        # if f.target_column_ind == f2.target_column_ind and f.src_column_ind == f1.src_column_ind:
        #     if f.source_val_indices[0] == f1.source_val_indices[0]:
        #         return True

        # filter if the distractor columns are irrelevent
        relevant_columns = {f1.src_column_ind, f2.src_column_ind, f2.target_column_ind}
        if not {f.src_column_ind, f.target_column_ind}.intersection(relevant_columns):
            return True

        # filter if the distractor is about the relevant rows
        if set(f.source_val_indices).intersection({f1.source_val_indices[0]}):
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
        random.seed(42)

        composition_questions = []

        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']
            none_ratio = self._qgen_config['templates'][0]['none_ratio']

            # generate composition questions by looping the facts
            for f1 in facts:
                # look for facts with one source
                if len(f1.source_val_indices) == 1:
                    source_column = f1.src_column_ind
                    target_column = f1.target_column_ind
                    for f2 in facts:
                        if (f2.src_column_ind == target_column) and (f2.target_column_ind != source_column) \
                                and f1.source_val_indices == f2.source_val_indices:
                            phrase = template
                            phrase = phrase.replace("[page_title]", f1.page_title)
                            phrase = phrase.replace("[table_title]", f1.table_title.strip())
                            phrase = phrase.replace("[source_column]", f1.src_column_header.strip())
                            phrase = phrase.replace("[target_column]", f2.target_column_header.strip())
                            phrase = phrase.replace("[from_cell_text]", str(f1.src_column_value).strip())

                            # sample distractors
                            possible_distractors = self.sample_distractors(facts, f1, f2, 'composition')
                            num_distractors = min(len(possible_distractors), 8)
                            distractors = random.sample(possible_distractors, num_distractors)

                            m = hashlib.md5()
                            m.update(context.id.encode())
                            m.update(str(source_column).encode())
                            m.update(str(target_column).encode())
                            m.update(f1.src_column_value.encode())
                            qid = 'SC-' + m.hexdigest()

                            # answer
                            answer = f2.target_column_values
                            # randomly downsample facts
                            question_facts = []
                            for q_fact in [f1, f2]:
                                if random.random() >= none_ratio:
                                    question_facts.append(q_fact.format_fact())
                                else:
                                    answer = ["none"]

                            num_facts = len(question_facts)
                            question_distractors = [d.format_fact() for d in
                                                    distractors[num_facts:]]

                            composition_questions.append(SyntheticQuestion(qid=qid,
                                                                           question=phrase,
                                                                           answers=answer,
                                                                           facts=question_facts,
                                                                           distractors=question_distractors,
                                                                           metadata={'type': 'composition',
                                                                                     'reasoning': self.reasoning_types,
                                                                                     'answer_type': 'entity',
                                                                                     'reversed_facts': [],
                                                                                     'template': f'Composition_1_hop',
                                                                                     }
                                                                           ))

        if len(composition_questions) > 2:
            composition_questions = random.sample(composition_questions, 2)
        return composition_questions
