import hashlib
import random
import json, re
import logging
from enum import Enum
from ExampleGeneration.common.table_wrapper import WikiTable
from ExampleGeneration.question_generators.question_generator import QuestionGenerator
from ExampleGeneration.common.multiqa_format_wrapper import Question, SyntheticQuestion


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
sample = []
question_index = {'i': 0}

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TemporalRelations(Enum):
    before = 0
    after = 1
    during = 2


class DateTimeRange:
    """
    date range object, we will keep the max and values for a range
    """

    def __init__(self, parsed_dates):
        parsed_to_unparsed_map = {val['parsed_value']: val['value']
                                  for val in parsed_dates}
        self.max_date_parsed = max(parsed_to_unparsed_map.keys())
        self.min_date_parsed = min(parsed_to_unparsed_map.keys())
        self.max_date = parsed_to_unparsed_map[self.max_date_parsed]
        self.min_date = parsed_to_unparsed_map[self.min_date_parsed]


class TemporalDifference(QuestionGenerator):
    def __init__(self, qgen_config, args):
        self.qgen_name = 'TemporalRange'
        self.reasoning_types = ['temporal_span']
        super().__init__(args)
        self._qgen_config = qgen_config
        self.temporal = True
        self.temporal_phrases = {'before': 'before',
                                 'after': 'after',
                                 'during': 'during the time'}

    def get_temporal_relation(self, DateTimeRange_1, DateTimeRange_2):
        """
        return the true temporal relation between the ranges
        """
        # check if the first event after before the second event
        if DateTimeRange_1.max_date_parsed < DateTimeRange_2.min_date_parsed:
            return TemporalRelations.before

        # check if the first event after after the second event
        if DateTimeRange_1.min_date_parsed > DateTimeRange_2.max_date_parsed:
            return TemporalRelations.after

        # check if the first event happened during the second event
        if DateTimeRange_1.min_date_parsed > DateTimeRange_2.min_date_parsed \
                and DateTimeRange_1.max_date_parsed < DateTimeRange_2.max_date_parsed:
            return TemporalRelations.during

        # otherwise return None
        return None

    def filter_temporal_distractor(self, f, f1, f2):
        """
        check if f can be a distractor for a compoisiton question from f1 to f2
        """
        # allow any distractor from the source to target columns
        if f.target_column_ind == f1.target_column_ind:
            if (f.source_val_indices != f1.source_val_indices) and (f.source_val_indices != f2.source_val_indices):
                return False
        return True

    def get_datetime_range(self, f):
        """
        get a DateTimeRange object for a specific fact
        """
        # first, get the parsed values
        f_target_column_values_parsed = [{'value': val,
                                          'parsed_value': self.parse_value(val,
                                                                           is_temporal=self.temporal)
                                          }
                                         for val in f.target_column_values]

        # if we the values were parsed, we can get the temporal relation
        f_range = DateTimeRange(f_target_column_values_parsed)
        return f_range

    def format_distractor(self, d, sorted_values_dict):
        """
        helper method for formatting distractors
        """
        # check if the distractor is a span
        if self.verify_sorted_values(d.target_column_values, sorted_values_dict) \
                and self.verify_values_in_following_rows(d.source_val_indices):

            # if so, get and format the range
            distractor_datetime_range = self.get_datetime_range(d)
            return d.format_datetime_fact(distractor_datetime_range)

        # else, use regular formatting
        else:
            return d.format_fact()

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

    def verify_values_in_following_rows(self, rows):
        """
        traverse the rows and verify that they follow each other in the table
        the distance between every row must be 1
        """
        curr_row = rows[0]
        for r in rows[1:]:
            if r - curr_row != 1:
                return False
            curr_row = r
        return True

    def verify_sorted_values(self, values, sorted_value_dict):
        """
        traverse the values and verify that no two following values have a rank difference of more than 1
        """
        curr_val = values[0]
        for val in values[1:]:
            # for every consecutive values, check their rank difference and filter if it is larger than 1
            if abs(sorted_value_dict[val] - sorted_value_dict[curr_val]) > 1:
                return False
            curr_val = val
        return True

    def generate(self, context):
        """
        :param from_question:
        :param to_question:
        :return: the composition question returned by injection of the first question to the second
        """
        # Generate facts
        table = WikiTable(context)
        facts = self.generate_facts(table)

        arithmetic_columns = table.get_arithmetic_inds()
        date_columns = [col for col in arithmetic_columns
                        if self.filter_date_columns(table, col)]

        # filter facts to facts about arithmetic columns
        facts = [f for f in facts
                 if f.target_column_ind in date_columns]

        template = self._qgen_config['templates'][0]['question_template']

        random.seed(42)

        # generate composition questions by looping the facts
        temporal_questions = []

        for date_column in date_columns:
            date_column_facts = [f for f in facts
                                 if f.target_column_ind == date_column]
            date_column_vals = {val for f in date_column_facts
                                for val in f.target_column_values}

            filter = False
            try:
                # let's verify that we can parse all the values
                # afterwards, we will sort all the datetime values
                parsed_values = [{'value': val, 'parsed_value': self.parse_value(val, is_temporal=self.temporal)}
                                 for val in date_column_vals]
            except:
                filter = True

            if not filter:
                sorted_values_dict = {key['value']: rank for rank, key in
                                      enumerate(sorted(parsed_values, key=lambda x: x['parsed_value']), 1)}

                for f1 in date_column_facts:

                    # we want the fact target to be in consecutive rows
                    if self.verify_sorted_values(f1.target_column_values, sorted_values_dict) \
                            and self.verify_values_in_following_rows(f1.source_val_indices):

                        for f2 in date_column_facts:

                            # look for facts with more than one source in following rows
                            if f1.source_val_indices != f2.source_val_indices \
                                    and len(f2.source_val_indices) > 1 \
                                    and self.verify_sorted_values(f2.target_column_values, sorted_values_dict) \
                                    and self.verify_values_in_following_rows(f2.source_val_indices):

                                # because the values were parsed, we can get the temporal relation
                                f1_range = self.get_datetime_range(f1)
                                f2_range = self.get_datetime_range(f2)

                                gold_temporal_relation = self.get_temporal_relation(f1_range, f2_range)

                                if gold_temporal_relation is not None:
                                    for temporal_relation in TemporalRelations:
                                        phrase = template
                                        phrase = phrase.replace("[page_title]", f1.page_title)
                                        phrase = phrase.replace("[table_title]", f1.table_title.strip())
                                        phrase = phrase.replace("[source_column1]", f1.src_column_header.strip())
                                        phrase = phrase.replace("[source_column2]", f2.src_column_header.strip())

                                        # get the phrase for the temporal relation
                                        phrase = phrase.replace("[COMPARATOR]",
                                                                self.temporal_phrases[temporal_relation.name])
                                        phrase = phrase.replace("[val_1]", f1.src_column_value.strip())
                                        phrase = phrase.replace("[val_2]", f2.src_column_value.strip())

                                        # sample distractors
                                        possible_distractors = self.sample_distractors(facts, f1, f2, 'temporal')
                                        num_distractors = min(len(possible_distractors), 4)
                                        distractors = random.sample(possible_distractors, num_distractors)

                                        # calculate qid
                                        m = hashlib.md5()
                                        m.update(context.id.encode())
                                        m.update(str(date_column).encode())
                                        m.update(str(f2.src_column_ind).encode())
                                        m.update(f1.src_column_value.encode())
                                        m.update(f2.src_column_value.encode())
                                        m.update('temporal'.encode())

                                        qid = 'TmpRel-' + m.hexdigest()
                                        # calculate answer
                                        answer = 'yes' if temporal_relation == gold_temporal_relation else 'no'

                                        temporal_questions.append(SyntheticQuestion(qid=qid,
                                                                                    question=phrase,
                                                                                    answers=[answer],
                                                                                    facts=[f1.format_datetime_fact(
                                                                                        f1_range),
                                                                                        f2.format_datetime_fact(
                                                                                            f2_range)],
                                                                                    distractors=[
                                                                                        self.format_distractor(d,
                                                                                                               sorted_values_dict)
                                                                                        for d in distractors],
                                                                                    metadata={
                                                                                        'type': f'temporal_span(gold:{gold_temporal_relation.name})',
                                                                                        'reasoning': self.reasoning_types,
                                                                                        'answer_type': 'boolean'}
                                                                                    )
                                                                  )

        # we want to sample equally between the different types
        # traverse the different quetion types
        sampled_questions = []
        for gold_relation in TemporalRelations:
            questions_for_type = [q for q in temporal_questions
                                  if q.metadata[
                                      'type'] == f'temporal_span(gold:{gold_relation.name})']

            # if there is more than one question for the type, sample just 1
            if len(questions_for_type) > 1:
                sampled_questions += random.sample(questions_for_type, 1)

        return sampled_questions
