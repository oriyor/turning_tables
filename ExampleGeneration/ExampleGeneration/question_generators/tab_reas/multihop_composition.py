import hashlib
import random
import logging
from ExampleGeneration.common.table_wrapper import WikiTable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sample = []
question_index = {'i': 0}
from ExampleGeneration.question_generators.question_generator import QuestionGenerator
from ExampleGeneration.common.multiqa_format_wrapper import Question, SyntheticQuestion

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ComplexCompositionFact():
    """
    a complex composition fact, that includes multiple facts
    """

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.intermediate_column_ind = f2.src_column_ind
        self.src_column_ind = f1.src_column_ind
        self.target_column_ind = f2.target_column_ind

    def format_fact(self):
        return [self.f1.format_fact(), self.f2.format_fact()]


class MultihopComposition(QuestionGenerator):
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

        # create complex facts with two terms
        complex_facts = []
        for f1_ind, f1 in enumerate(facts):
            if len(f1.source_val_indices) == 1:

                for f2 in facts[f1_ind:]:
                    if (f2.src_column_ind == f1.target_column_ind) and (
                            f2.target_column_ind != f1.src_column_ind) and f1.source_val_indices == f2.source_val_indices:
                        complex_facts.append(ComplexCompositionFact(f1, f2))

        composition_questions = []

        for config_template in self._qgen_config['templates']:

            template = config_template['question_template']

            # generate composition questions by looping the facts
            for f in facts:

                for cf in complex_facts:
                    if f.source_val_indices == cf.f1.source_val_indices:
                        if f.src_column_ind == cf.f2.target_column_ind:
                            if f.target_column_ind not in [cf.intermediate_column_ind, cf.src_column_ind,
                                                           cf.target_column_ind]:
                                phrase = template
                                phrase = phrase.replace("[page_title]", f.page_title)
                                phrase = phrase.replace("[table_title]", f.table_title.strip())
                                phrase = phrase.replace("[source_column]", cf.f1.src_column_header.strip())
                                phrase = phrase.replace("[target_column]", f.target_column_header.strip())
                                phrase = phrase.replace("[from_cell_text]", str(cf.f1.src_column_value).strip())

                                # sample distractors
                                # this is a special case in which we'll look for distractors for each of the 3 hops

                                distractors = []
                                for fact_to_find_distractors in [cf.f1, cf.f2, f]:
                                    fact_distractors = [f_distractor for f_distractor in facts
                                                        if
                                                        f_distractor.src_column_ind == fact_to_find_distractors.src_column_ind
                                                        and f_distractor.target_column_ind == fact_to_find_distractors.target_column_ind
                                                        and len(f_distractor.source_val_indices) == 1
                                                        and f_distractor.source_val_indices != fact_to_find_distractors.source_val_indices]

                                    num_distractors = min(len(fact_distractors), 4)
                                    distractors += random.sample(fact_distractors, num_distractors)
                                question_distractors = [d.format_fact() for d in
                                                        distractors]

                                m = hashlib.md5()
                                m.update(context.id.encode())
                                m.update(str(cf.src_column_ind).encode())
                                m.update(str(cf.intermediate_column_ind).encode())
                                m.update(str(cf.target_column_ind).encode())
                                m.update(str(f.target_column_ind).encode())
                                m.update(cf.f1.src_column_value.encode())
                                qid = 'MHC-' + m.hexdigest()

                                # answer
                                answer = f.target_column_values

                                composition_questions.append(SyntheticQuestion(qid=qid,
                                                                               question=phrase,
                                                                               answers=answer,
                                                                               facts=[f.format_fact(),
                                                                                      cf.f1.format_fact(),
                                                                                      cf.f2.format_fact()],
                                                                               distractors=question_distractors,
                                                                               metadata={'type': 'composition_2_hop',
                                                                                         'reasoning': self.reasoning_types,
                                                                                         'answer_type': 'entity',
                                                                                         'reversed_facts': [],
                                                                                         'template': f'composition_2_hop',
                                                                                         }

                                                                               ))

        if len(composition_questions) > 6:
            composition_questions = random.sample(composition_questions, 6)
        return composition_questions
