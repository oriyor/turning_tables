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


class Arithmetic(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'arithmetic'
        self.reasoning_types = ['arithmetic']
        super().__init__(args)
        self._qgen_config = qgen_config
        self.temporal = False
        self.arithmetic_operations = {'highest': {'op': max, 'filter_duplicate_facts': True},
                                      'lowest': {'op': min, 'filter_duplicate_facts': True},
                                      'sum of the': {'op': sum, 'filter_duplicate_facts': False},
                                      'total number of': {'op': sum, 'filter_duplicate_facts': False}}

        self.columns_blacklist = ['number', 'week', 'order']

    def filter_arithmetic_distractor(self, f, f1, k):
        """
        check if f can be a distractor for an arithmetic question about f1
        """

        # allow any to or from the brdige
        if (f.target_column_ind == k and f.src_column_ind == f1.src_column_ind) \
                or (f.target_column_ind == f1.target_column_ind and f.src_column_ind == k)\
                or (f.target_column_ind == f1.target_column_ind and f.src_column_ind == f1.src_column_ind):

            # we don't want any interescting facts
            if not set(f.source_val_indices).intersection(set(f1.source_val_indices)):
                return False
        return True

    def filter_key_column(self, table, src_column_id):
        """
        keep only source columns from key columns
        """
        if src_column_id in table.get_key_column() \
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

        random.seed(42)

        # todo move this to classification
        arithmetic_columns = table.get_arithmetic_inds()
        numeric_columns = [col for col in arithmetic_columns
                           if not self.filter_date_columns(table, col)]

        # generate composition questions by looping the facts
        key_columns_all = table.get_key_column()
        entities_columns = table.get_entities_columns()
        key_columns = key_columns_all.union(entities_columns)

        # init fields
        numeric_facts = [f for f in facts
                         if f.target_column_ind in numeric_columns
                         ]
        arithmetic_questions = []

        # iterate all templates
        for template_config in self._qgen_config['templates']:
            template = template_config['question_template']

            # generate composition questions by looping the facts
            for f in numeric_facts:
                # look for facts with more than one source
                if len(f.source_val_indices) > 1 \
                        and f.src_column_ind not in arithmetic_columns:
                    source_column = f.src_column_ind
                    target_column = f.target_column_ind

                    try:
                        target_vals = [self.parse_value(val, self.temporal)
                                       for val in f.target_column_values]

                    except:
                        continue

                    # todo remove to config
                    if f.formatted_target_column_header not in self.columns_blacklist:

                        # self.generate_facts(table)
                        for op in self.arithmetic_operations:

                            filter_duplicate_facts = self.arithmetic_operations[op]['filter_duplicate_facts']
                            phrase = template
                            phrase = phrase.replace("[page_title]", f.page_title)
                            phrase = phrase.replace("[table_title]", f.table_title.strip())
                            phrase = phrase.replace("[source_column]", f.formatted_src_column_header.strip())
                            phrase = phrase.replace("[target_column]", f.formatted_target_column_header.strip())
                            phrase = phrase.replace("[ARITHMETIC_OPERATOR]", op)
                            phrase = phrase.replace("[from_cell_text]", f.src_column_value.strip())

                            m = hashlib.md5()
                            m.update(context.id.encode())
                            m.update(str(source_column).encode())
                            m.update(str(target_column).encode())
                            m.update(f.src_column_value.encode())
                            qid = 'Arithmetic-' + m.hexdigest()

                            # find answer, filter if no suitable answer exists

                            answer = self.arithmetic_operations[op]['op'](target_vals)

                            # look for supporting facts, this doesn't effect the question, only the facts
                            for key_column in key_columns:

                                # look for facts from the key column to the target column
                                key_column_facts = [f1.format_fact() for f1 in facts
                                                    if f1.src_column_ind == key_column
                                                    and f1.target_column_ind == f.target_column_ind
                                                    and len(f1.source_val_indices) == 1
                                                    and f1.source_val_indices[0] in f.source_val_indices]

                                if len(key_column_facts) == len(f.source_val_indices) \
                                        and len(key_column_facts) > 1:
                                    # find the bridge fact between the target column and the key column
                                    bridge_fact = {f1.format_fact() for f1 in facts
                                                   if
                                                   f1.target_column_ind == key_column and f1.src_column_ind == f.src_column_ind
                                                   and f1.source_val_indices == f.source_val_indices}

                                    # change answer to int if possible
                                    if type(answer) != int and answer.is_integer():
                                        answer = int(answer)

                                    # sample distractors
                                    possible_distractors = self.sample_distractors(facts, f, key_column, 'arithmetic')
                                    num_distractors = min(len(possible_distractors), 6)
                                    distractors = random.sample(possible_distractors, num_distractors)

                                    # we will use reverse facts with equal probability
                                    f_reversed = key_column_facts \
                                                + list(
                                                    bridge_fact)

                                    reverse_facts = random.choice([True, False])

                                    if reverse_facts:
                                        question_facts = f_reversed
                                        question_template = f'arithmetic-{op}-reverse-facts'

                                    else:
                                        # here we want to not filter duplicates for sum/average
                                        question_facts = [f.format_fact(filter_duplicate_facts=filter_duplicate_facts)]
                                        question_template = f'arithmetic-{op}'

                                    arithmetic_questions.append(SyntheticQuestion(qid=qid,
                                                                                  question=phrase,
                                                                                  answers=[answer],
                                                                                  facts=question_facts,
                                                                                  distractors=[d.format_fact() for d in
                                                                                               distractors],
                                                                                  metadata={'type': 'arithmetic',
                                                                                            'reasoning': self.reasoning_types,
                                                                                            'answer_type': 'numeric',
                                                                                            'template': question_template,
                                                                                            'reversed_facts': f_reversed,
                                                                                            }
                                                                                  )
                                                                )

        # we want to sample equally between the different types
        # traverse the different quetion types
        sampled_questions = []
        for gold_op in self.arithmetic_operations:
            questions_for_type = [q for q in arithmetic_questions
                                  if f'arithmetic-{gold_op}'
                                  in q.metadata['template']]

            # if there is more than one question for the type, sample just 1
            if len(questions_for_type) > 10:
                sampled_questions += random.sample(questions_for_type, 10)

        return sampled_questions
