import hashlib
import random
import json, re
import logging
from ExampleGeneration.question_generators.question_generator import QuestionGenerator
from ExampleGeneration.common.multiqa_format_wrapper import Question, SyntheticQuestion
from ExampleGeneration.common.table_wrapper import WikiTable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sample = []
question_index = {'i': 0}


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NumericBooleanComparison(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'BooleanComparison'
        self.reasoning_types = ['comparison']
        super().__init__(args)
        self._qgen_config = qgen_config

    def filter_comparison_distractor(self, f, f1, f2):
        """
        check if f can be a distractor for a compoisiton question from f1 to f2
        """
        # allow any numeric distractor
        if f.src_column_ind == f1.src_column_ind and f.src_column_ind == f2.src_column_ind:

            # if the target column of the distractor is similar to the gold one, make sure we don't repeat indices
            # if the indices are different, we can use the distractor
            if (f.source_val_indices != f1.source_val_indices) and (f.source_val_indices != f2.source_val_indices):
                return False

            # otherwise, we can only use if the target column is different
            if f.target_column_ind != f1.target_column_ind:
                return False

        return True

    def filter_date_columns(self, table, column_id):
        # checking if the column is date:
        target_column_header = table.table.header[column_id]
        if re.search(r'year|years|date', target_column_header.column_name, re.IGNORECASE) is not None \
                or ('DATE' in target_column_header.metadata['ner_appearances_map'] and \
                    target_column_header.metadata['ner_appearances_map']['DATE'] / \
                    len(target_column_header.metadata['parsed_values']) > 0.6):
            return True
        else:
            return False

    def filter_key_column(self, table, src_column_id, key_columns):
        """
        keep only source columns from key columns
        """
        # todo compare get_entities_columns and get_key_column, see which one is in one or not the other
        if src_column_id in key_columns\
                and src_column_id not in table.get_arithmetic_inds():
            return False
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

        key_columns_all = table.get_key_column()
        entities_columns = table.get_entities_columns()
        key_columns = key_columns_all.union(entities_columns)

        # todo move this to classification
        arithmetic_columns = table.get_arithmetic_inds()
        numeric_columns = [col for col in arithmetic_columns
                           if not self.filter_date_columns(table, col)]

        # filter facts to facts about arithmetic columns
        facts = [f for f in facts
                 if f.target_column_ind in numeric_columns]
        comparison_questions = []

        random.seed(42)

        # traverse all templates
        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']
            higher_comparator_singular = template_config['numeric_higher_comparator_singular']
            lower_comparator_singular = template_config['numeric_lower_comparator_singular']
            higher_comparator_plural = template_config['numeric_higher_comparator_plural']
            lower_comparator_plural = template_config['numeric_lower_comparator_plural']
            none_ratio = template_config['none_ratio']

            # generate composition questions by looping the facts
            for fact_ind, f1 in enumerate(facts):

                # look for facts with one source

                if len(f1.source_val_indices) == 1:
                    source_column = f1.src_column_ind
                    target_column = f1.target_column_ind

                    # check if we can use f2
                    # we only need to traverse pairs we haven't seen
                    for f2 in facts[fact_ind:]:
                        if (f2.target_column_ind == target_column) and (f2.src_column_ind == source_column) \
                                and f1.source_val_indices != f2.source_val_indices \
                                and len(f1.source_val_indices) == len(f2.source_val_indices) == 1 \
                                and not self.filter_key_column(table, f1.src_column_ind, key_columns):

                            # parse the fact
                            filter = False
                            try:
                                f1_parsed_values = [self.parse_value(val, is_temporal=False)
                                                    for val in f1.target_column_values]
                                f2_parsed_values = [self.parse_value(val, is_temporal=False)
                                                    for val in f2.target_column_values]
                            except:
                                filter = True

                            # make sure the formatted header can be used to determine if the target is plural or singular
                            target_column_header = f2.formatted_target_column_header.strip()

                            if not filter and len(target_column_header):

                                # replace the comparator
                                use_higher_comparator = random.choice([False, True])

                                # check if the column is singular or plural
                                # we use a heuristic approach: if the target column ends with an s, and does not of a number
                                # of prefix, it is plural

                                if target_column_header[-1] != 's' \
                                        or target_column_header.startswith('#') \
                                        or target_column_header.startswith('number'):
                                    str_comparator = higher_comparator_singular if use_higher_comparator else lower_comparator_singular

                                else:
                                    str_comparator = higher_comparator_plural if use_higher_comparator else lower_comparator_plural

                                phrase = template
                                phrase = phrase.replace("[page_title]", f1.page_title)
                                phrase = phrase.replace("[table_title]", f1.table_title.strip())
                                phrase = phrase.replace("[source_column]", f1.formatted_src_column_header.strip())
                                phrase = phrase.replace("[target_column]", target_column_header)
                                phrase = phrase.replace("[COMPARATOR]", str_comparator)
                                phrase = phrase.replace("[val_1]", f1.src_column_value.strip())
                                phrase = phrase.replace("[val_2]", f2.src_column_value.strip())

                                # sample distractors
                                possible_distractors = self.sample_distractors(facts, f1, f2, 'comparison')
                                num_distractors = min(len(possible_distractors), 8)
                                distractors = random.sample(possible_distractors, num_distractors)

                                m = hashlib.md5()
                                m.update(context.id.encode())
                                m.update(str(source_column).encode())
                                m.update(str(target_column).encode())
                                m.update(f1.src_column_value.encode())
                                qid = 'NumCMPBool-' + m.hexdigest()

                                # find answer, filter if no suitable answer exists
                                question_filter = False
                                answer = ""
                                if use_higher_comparator:
                                    max_f1 = max(f1_parsed_values)
                                    max_f2 = max(f2_parsed_values)
                                    if max_f1 == max_f2:
                                        question_filter = True
                                    else:
                                        if max_f1 > max_f2:
                                            delta = max_f1 - max_f2
                                            answer = "yes"
                                        else:
                                            delta = max_f2 - max_f1
                                            answer = "no"

                                else:
                                    min_f1 = min(f1_parsed_values)
                                    min_f2 = min(f2_parsed_values)
                                    if min_f1 == min_f2:
                                        question_filter = True
                                    else:
                                        if min_f1 < min_f2:
                                            delta = min_f2 - min_f1
                                            answer = "yes"
                                        else:
                                            delta = min_f1 - min_f2
                                            answer = "no"

                                # verify source values aren't numeric
                                for src_val in [f1.src_column_value, f2.src_column_value]:
                                    src_val = src_val.replace('+', '').replace(',', '').replace('-', '').replace(':', '') \
                                        .replace('–', '').replace('-', '').strip()
                                    try:
                                        if float(src_val):
                                            question_filter = True
                                    except:
                                        continue

                                if not question_filter:

                                    # add delta to the answer to increase supervision
                                    answer = f'{answer}'

                                    # randomly downsample facts
                                    question_facts = []
                                    for q_fact in [f1, f2]:
                                        if random.random() >= none_ratio:
                                            question_facts.append(q_fact.format_fact())
                                        else:
                                            answer = "none"

                                    num_facts = len(question_facts)
                                    question_distractors = [d.format_fact() for d in
                                                            distractors[num_facts:]]

                                    comparison_questions.append(SyntheticQuestion(qid=qid,
                                                                                  question=phrase,
                                                                                  answers=[answer],
                                                                                  facts=question_facts,
                                                                                  distractors=question_distractors,
                                                                                  metadata={'type': 'numeric_comparison_boolean',
                                                                                            'reasoning': self.reasoning_types,
                                                                                            'answer_type': 'bool',
                                                                                            'reversed_facts': [],
                                                                                            'template': template_config[
                                                                                                          'name']
                                                                                            }
                                                                                  )
                                                                )

        if len(comparison_questions) > 10:
            comparison_questions = random.sample(comparison_questions, 10)
        return comparison_questions
