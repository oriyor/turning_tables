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


class Simple(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'Simple'
        self.reasoning_types = ['Simple']
        super().__init__(args)
        self._qgen_config = qgen_config

    def filter_simple_distractor(self, f, f1):
        """
        check if f can be a distractor for a compoisiton question from f1 to f2
        """

        # # filter if this distractor is the explicit answer to the question
        # if f.target_column_ind == f2.target_column_ind and f.src_column_ind == f1.src_column_ind:
        #     if f.source_val_indices[0] == f1.source_val_indices[0]:
        #         return True

        # filter if the distractor columns are irrelevent
        relevant_columns = {f1.src_column_ind, f1.target_column_ind}
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

        simple_questions = []

        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']
            none_ratio = self._qgen_config['templates'][0]['none_ratio']

            # generate composition questions by looping the facts
            for f1 in facts:
                # look for facts with one source
                if len(f1.source_val_indices) == 1:
                    source_column = f1.src_column_ind
                    target_column = f1.target_column_ind

                    phrase = template
                    phrase = phrase.replace("[page_title]", f1.page_title)
                    phrase = phrase.replace("[table_title]", f1.table_title.strip())
                    phrase = phrase.replace("[source_column]", f1.src_column_header.strip())
                    phrase = phrase.replace("[target_column]", f1.target_column_header.strip())
                    phrase = phrase.replace("[from_cell_text]", str(f1.src_column_value).strip())

                    # sample distractors
                    possible_distractors = self.sample_distractors(facts, f1, f1, 'simple')
                    num_distractors = min(len(possible_distractors), 8)
                    distractors = random.sample(possible_distractors, num_distractors)

                    m = hashlib.md5()
                    m.update(context.id.encode())
                    m.update(str(source_column).encode())
                    m.update(str(target_column).encode())
                    m.update(f1.src_column_value.encode())
                    qid = 'Simple-' + m.hexdigest()

                    # answer
                    answer = f1.target_column_values
                    # randomly downsample facts
                    question_facts = [f1.format_fact()]

                    num_facts = len(question_facts)
                    question_distractors = [d.format_fact() for d in
                                            distractors[num_facts:]]

                    simple_questions.append(SyntheticQuestion(qid=qid,
                                                                   question=phrase,
                                                                   answers=answer,
                                                                   facts=question_facts,
                                                                   distractors=question_distractors,
                                                                   metadata={'type': 'simple',
                                                                             'reasoning': self.reasoning_types,
                                                                             'answer_type': 'entity',
                                                                             'reversed_facts': [],
                                                                             'template': f'somple',
                                                                             }
                                                                   ))

        if len(simple_questions) > 1:
            simple_questions = random.sample(simple_questions, 1)
        return simple_questions
