import hashlib
import random
import json, re
import logging
from copy import copy
from dateutil import parser
from ExampleGeneration.common.table_wrapper import WikiTable
from dateutil.relativedelta import relativedelta
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sample = []
question_index = {'i': 0}
from ExampleGeneration.question_generators.question_generator import QuestionGenerator
from ExampleGeneration.common.multiqa_format_wrapper import Question, SyntheticQuestion

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TemporalBooleanComparison(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'TemporalBooleanComparison'
        self.reasoning_types = ['temporal_events']
        super().__init__(args)
        self._qgen_config = qgen_config
        self.source_column_blacklist = 'date'
        self.sample_examples_per_context = qgen_config['sample_examples_per_context']

    def filter_temporal_distractor(self, f, f1, f2):
        """
        check if f can be a distractor for a compoisiton question from f1 to f2
        """
        # allow any distractor from the source to target columns
        if f.target_column_ind == f1.target_column_ind:
            if (f.source_val_indices != f1.source_val_indices) and (f.source_val_indices != f2.source_val_indices):
                return False
        return True

    def find_how_many_comparator(self, rd):
        """
        if we want to ask a how many question, we need to the difference to be only years, months, or days
        """
        if np.sum([rd.years != 0, rd.months != 0, rd.days != 0]) > 1:
            return None
        elif rd.years != 0:
            return 'years'
        elif rd.months != 0:
            return 'months'
        elif rd.days != 0:
            return 'days'

    def prettify_relative_date(self, rd):
        """
        prettify the relative date time delta
        """
        years = f'{rd.years} year' if rd.years != 0 else ''
        month_prefix = ', ' if len(years) else ''
        months = f'{month_prefix}{rd.months} month' if rd.months != 0 else ''
        days_prefix = ', and ' if (len(months) or (len(years) and months == '')) else ''
        days = f'{days_prefix}{rd.days} day' if rd.days != 0 else ''

        # add suffices
        years_suffix = 's' if rd.years not in [-1, 0, 1] else ''
        months_suffix = 's' if rd.months not in [-1, 0, 1] else ''
        days_suffix = 's' if rd.days not in [-1, 0, 1] else ''

        # remove minus signs
        return f'{years}{years_suffix}{months}{months_suffix}{days}{days_suffix}'.replace('-', '')

    def filter_date_columns(self, table, column_id):
        # checking if the column is date:
        target_column_header = table.table.header[column_id]
        if 'type' in target_column_header.metadata and target_column_header.metadata['type'] == 'Datetime':
            return True
        if re.search(r'year|years|date', target_column_header.column_name, re.IGNORECASE) is not None \
                or ('DATE' in target_column_header.metadata['ner_appearances_map'] and \
                    target_column_header.metadata['ner_appearances_map']['DATE'] / \
                    len(target_column_header.metadata['parsed_values']) > 0.6):
            return True
        else:
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

        # todo move this to classification
        arithmetic_columns = table.get_arithmetic_inds()
        date_columns = [col for col in arithmetic_columns
                        if self.filter_date_columns(table, col)]

        # filter facts to facts about arithmetic columns
        facts = [f for f in facts
                 if f.target_column_ind in date_columns
                 and f.src_column_ind not in date_columns
                 and not re.search(self.source_column_blacklist, f.formatted_src_column_header, re.IGNORECASE)]

        # filter tables with more than one date column
        if len(date_columns) > 1:
            facts = []

        random.seed(42)

        temporal_questions = []

        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']
            higher_comparator = template_config['numeric_higher_comparator']
            lower_comparator = template_config['numeric_lower_comparator']
            none_ratio = self._qgen_config['none_ratio']

            if template_config['enable']:

                # generate tempotal questions by looping the facts
                for f1 in facts:
                    # look for facts with one source
                    if len(f1.source_val_indices) == 1:
                        source_column = f1.src_column_ind
                        target_column = f1.target_column_ind
                        for f2 in facts:
                            if (f2.target_column_ind == target_column) \
                                    and f1.source_val_indices != f2.source_val_indices \
                                    and len(f1.source_val_indices) == len(f2.source_val_indices) == 1:

                                # replace the comparator
                                use_higher_comparator = random.choice([False, True])
                                str_comparator = higher_comparator if use_higher_comparator else lower_comparator

                                phrase = template
                                phrase = phrase.replace("[page_title]", f1.page_title)
                                phrase = phrase.replace("[table_title]", f1.table_title.strip())
                                phrase = phrase.replace("[source_column1]", f1.src_column_header.strip())
                                phrase = phrase.replace("[source_column2]", f2.src_column_header.strip())
                                phrase = phrase.replace("[target_column]", f1.target_column_header.strip())

                                phrase = phrase.replace("[COMPARATOR]", str_comparator)
                                phrase = phrase.replace("[val_1]", f1.src_column_value.strip())
                                phrase = phrase.replace("[val_2]", f2.src_column_value.strip())

                                # sample distractors
                                possible_distractors = self.sample_distractors(facts, f1, f2, 'temporal')
                                num_distractors = min(len(possible_distractors), 8)
                                distractors = random.sample(possible_distractors, num_distractors)

                                m = hashlib.md5()
                                m.update(context.id.encode())
                                m.update(str(source_column).encode())
                                m.update(str(target_column).encode())
                                m.update(f1.src_column_value.encode())
                                m.update('temporal'.encode())

                                qid = 'TMP-' + m.hexdigest()

                                # find answer, filter if no suitable answer exists
                                filter = False
                                answer = ""

                                # try parsing to datetime objects
                                try:
                                    f1_target_column_values_parsed = [self.parse_value(val, is_temporal=True) for val in
                                                                      f1.target_column_values]
                                    f2_target_column_values_parsed = [self.parse_value(val, is_temporal=True) for val in
                                                                      f2.target_column_values]

                                except:
                                    filter = True

                                # calculate answer
                                if not filter:
                                    answer_phrase = 'the [source_column] was [val]'

                                    if use_higher_comparator:
                                        max_f1 = max(f1_target_column_values_parsed)
                                        max_f2 = max(f2_target_column_values_parsed)
                                        if max_f1 == max_f2:
                                            filter = True

                                        else:
                                            rd = relativedelta(max_f1, max_f2)
                                            delta = self.prettify_relative_date(rd)
                                            if max_f1 > max_f2:
                                                answer_phrase = f'yes'
                                                # answer_phrase = answer_phrase.replace("[source_column]",
                                                #                                       f1.src_column_header.strip())
                                                # answer_phrase = answer_phrase.replace("[val]",
                                                #                                       f1.src_column_value.strip())

                                            else:

                                                answer_phrase = f'no'

                                    else:
                                        min_f1 = min(f1_target_column_values_parsed)
                                        min_f2 = min(f2_target_column_values_parsed)
                                        if min_f1 == min_f2:
                                            filter = True
                                        else:
                                            rd = relativedelta(min_f1, min_f2)
                                            delta = self.prettify_relative_date(rd)
                                            if min_f1 < min_f2:
                                                answer_phrase = f'yes'

                                            else:
                                                answer_phrase = f'no'


                                if not filter:

                                    if 'comparator' in template_config and template_config['comparator']:
                                        answer = f'{answer_phrase}'
                                        answer_type = 'bool'
                                        question_type = 'temporal_comparison_boolean'
                                    else:
                                        answer = delta
                                        answer_type = 'temporal'
                                        question_type = 'temporal_difference'


                                    # if we want to ask a how many filter, we need to check the datetime difference is valid
                                    if 'how_many' in template_config and template_config['how_many']:
                                        how_many_comparator = self.find_how_many_comparator(rd)

                                        if how_many_comparator is not None:
                                            phrase = phrase.replace("[TIME]", how_many_comparator)
                                        else:
                                            filter = True

                                    if not filter:

                                        # randomly downsample facts
                                        question_facts = []
                                        for q_fact in [f1, f2]:
                                            if random.random() >= none_ratio:
                                                question_facts.append(
                                                    q_fact.format_fact(date_time=random.choice([True, False])))
                                            else:
                                                answer = "none"

                                        num_facts = len(question_facts)
                                        question_distractors = [d.format_fact(random.choice([True, False])) for d in
                                                                distractors[num_facts:]]

                                        temporal_questions.append(SyntheticQuestion(qid=qid,
                                                                                    question=phrase,
                                                                                    answers=[answer],
                                                                                    facts=question_facts,
                                                                                    distractors=question_distractors,
                                                                                    facts_ids=[f1.fact_id, f2.fact_id],
                                                                                    metadata={'type': question_type,
                                                                                              'reasoning': self.reasoning_types,
                                                                                              'answer_type': 'answer_type',
                                                                                              'reversed_facts': [],
                                                                                              'template':
                                                                                                  template_config[
                                                                                                      'name']
                                                                                              }
                                                                                    )
                                                                  )

        if len(temporal_questions) > self.sample_examples_per_context:
            temporal_questions = random.sample(temporal_questions, self.sample_examples_per_context)

        return temporal_questions
