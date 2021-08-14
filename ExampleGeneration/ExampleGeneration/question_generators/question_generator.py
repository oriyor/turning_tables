import json
import logging
import re, gzip
import pandas as pd
import random
from ExampleGeneration.common.multiqa_format_wrapper import Answer, Fact
from dateutil import parser

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QuestionGenerator():
    def __init__(self, args):
        # For question matching, we use the config entry for gen_question_from_templates "match_annotated_questions"
        # if true, this will match only annotated questions, if false annotated questions are ignored.
        if 'match_annotated_questions' in args and args.match_annotated_questions:
            self._annotated_questions = args._annotated_questions
            # self.load_annotated_questions(args.annotated_questions_file)
        else:
            self._annotated_questions = None

        if 'base_working_directory' in args:
            self.base_working_directory = args.base_working_directory

        self.disallowed_cells_text = ['total'] + [f'rowspan="{i}" ' for i in range(15)]

    def Generate(self, context, args):
        pass

    def generate_facts(self, table):
        """
        :param context:
        :return: generate facts about the context
        """

        # we can generate the same fact at different ids, hence we will keep count of the fact ids we generated
        facts = []
        fact_ids = set()

        row_inds = table.get_row_inds()
        tar_col_inds = table.get_column_inds()
        src_col_inds = table.get_column_inds()
        for src_col in src_col_inds:
            src_col_header = table._table.header[src_col].column_name

            # we want to filter short cells only for textual cells
            filter_short_cells = 'parsed_values' in table._table.header[src_col].metadata

            for tar_col in tar_col_inds:
                tar_col_header = table._table.header[tar_col].column_name
                tar_col_numeric = 'float' in table._table.header[tar_col].metadata
                tar_col_datetime = 'Datetime' in table._table.header[tar_col].metadata

                if src_col != tar_col:
                    src_values_set = set()
                    # for every source and target column, we go over all rows and look for facts
                    for row in row_inds:
                        src_cell_val = table.table.table_rows[row][src_col].text
                        # a fact is defined by the source val
                        if src_cell_val not in src_values_set:
                            src_values_set.add(src_cell_val)

                        src_val_indices = [i for i in range(len(table._table.table_rows)) \
                                           if table._table.table_rows[i][src_col].text == src_cell_val]

                        target_values_at_src_indices = [table._table.table_rows[i][tar_col].text
                                                        for i in src_val_indices
                                                        if
                                                        not any(
                                                            filter_txt in table._table.table_rows[i][tar_col].text
                                                                for filter_txt in self.disallowed_cells_text)]

                        # filter bad facts
                        keep_flag = self.filter_fact(table, src_col, tar_col,
                                                     src_cell_val, target_values_at_src_indices,
                                                     filter_short_cells=filter_short_cells)

                        fact = Fact(table_title=table.table.table_name,
                                    page_title=table.page_title,
                                    page_url=table.page_url,
                                    src_column_ind=src_col,
                                    target_column_ind=tar_col,
                                    source_val_indices=src_val_indices,
                                    src_column_header=src_col_header,
                                    target_column_header=tar_col_header,
                                    src_column_value=src_cell_val,
                                    target_column_values=target_values_at_src_indices,
                                    is_numeric=tar_col_numeric,
                                    is_datetime=tar_col_datetime,
                                    filtered=not keep_flag)

                        # verify this is a new fact
                        if fact.fact_id not in fact_ids:

                            # append fact and keep score of fact id
                            facts.append(fact)
                            fact_ids.add(fact.fact_id)
        return facts

    def sample_distractors(self, facts, f1, f2, reasoning):
        """
        sample distractors from a facts list, and source and target facts f1 f2
        """
        if reasoning == 'composition':
            distractors_facts = [f for f in facts
                                 if not self.filter_compoisition_distractor(f, f1, f2)]
        if reasoning == 'comparison':
            distractors_facts = [f for f in facts
                                 if not self.filter_comparison_distractor(f, f1, f2)]
        if reasoning == 'temporal':
            distractors_facts = [f for f in facts
                                 if not self.filter_temporal_distractor(f, f1, f2)]
        if reasoning == 'arithmetic':
            distractors_facts = [f for f in facts
                                 if not self.filter_arithmetic_distractor(f, f1, f2)]
        if reasoning == 'counting':
            distractors_facts = [f for f in facts
                                 if not self.filter_counting_distractor(f, f1)]
        if reasoning == 'simple':
            distractors_facts = [f for f in facts
                                 if not self.filter_simple_distractor(f, f1)]

        return distractors_facts

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

    def is_date_column(self, table, column_id):
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

    def parse_value(self, value, is_temporal=False):
        """
        helper method to parse a datetime or numeric value
        """
        if is_temporal:
            return parser.parse(value, ignoretz=True)
        return float(value.replace(',', ''))

    def reverse_fact(self, fact, all_facts):
        """
        try and reverse the facts, such that if we have a fact that is a list we reverse it to a list of facts of size 1
        """
        # reverse the target and source columns
        reverse_target_column = fact.src_column_ind
        reverse_source_column = fact.target_column_ind

        # we are only interested in the source indices
        source_indices = fact.source_val_indices
        reversed_facts = [f for f in all_facts
                          if f.target_column_ind == reverse_target_column
                          and f.src_column_ind == reverse_source_column
                          ]

        # find every fact of length 1 between the reversed columns at the relevant indices
        reversed_facts_at_indices = [f for f in reversed_facts if len(f.source_val_indices) == 1
                                     and f.source_val_indices[0] in source_indices]

        # if we found all the relevant facts, let's return the reversed fact
        if len(reversed_facts_at_indices) == len(source_indices):
            return reversed_facts_at_indices

        # otherwise, return none
        return None

    def filter_fact(self, table, source_column, target_column,
                    src_cell_val, target_values_at_src_indices, filter_short_cells):
        """
        a tuple -
        1. Boolean validity flag that determines if it is possible to ask a
            property question from the source_cell to the target_column
        """

        if len(table._table.header[target_column].column_name) < 3 or \
                len(table._table.header[source_column].column_name) < 3 or \
                len(table._table.header[target_column].column_name) > 15 or \
                len(table._table.header[source_column].column_name) > 15 or \
                len(table.page_title) < 3 or len(table.page_title) > 30 or \
                len(table._table.table_name) < 3 or len(table._table.table_name) > 25 or \
                len(str(src_cell_val)) > 35:
            return False

        # we want to filter short textual cells, but not numeric
        if filter_short_cells and len(str(src_cell_val)) < 3:
            return False

        # if none of the target values are valid
        if len(target_values_at_src_indices) == 0:
            return False

        # valid cell values for fact
        if str(src_cell_val).lower() in [str(cell_val).lower()
                                         for cell_val in target_values_at_src_indices] or \
                table._table.header[source_column].column_name.lower().find(str(src_cell_val).lower()) > -1 or \
                table._table.header[target_column].column_name.lower().find(
                    str(target_values_at_src_indices[0]).lower()) > -1:
            return False

        # no empty strings in target column
        if len([str(val).lower() for val in target_values_at_src_indices
                if str(val) == ""]) > 0:
            return False

        # verify that the target cell is not unique in the column
        if len(target_values_at_src_indices) == table.m:
            return False

        # checking all cell values contain at least one number or letter
        for cell_val in target_values_at_src_indices + [src_cell_val]:
            if len(re.findall('[a-zA-Z0-9]', cell_val)) == 0:
                return False

        # do not support no answer or answers sets that contain 'Total' or '-'
        if target_values_at_src_indices is None \
                or len(target_values_at_src_indices) == 0 \
                or target_values_at_src_indices[0] == '' \
                or target_values_at_src_indices[0] == None \
                or [cell_val for cell_val in target_values_at_src_indices
                    if cell_val == 'Total' or cell_val == '-']:
            logger.debug(f'found empty answer or total or - in Fact: {target_values_at_src_indices}')
            return False

        # filter bad cells and images, similar to filter_table_question()
        if re.search('(N/A|[0-9]+px)|nbsp|px|/|\n', src_cell_val) or 'total' in src_cell_val.lower():
            return False

        # filter bad answer cells
        for cell_val in target_values_at_src_indices:
            if re.search('(N/A|[0-9]+px)|nbsp|px|/|\n', cell_val):
                return False

        # filter facts to image columns
        image_columns = {table.table.header[col].metadata['descriptive_image_column']
                         for col in range(table.n)
                         if 'descriptive_image_column' in table.table.header[col].metadata}
        if target_column in image_columns:
            return False

        # filter disallowed source cells
        if any(filter_txt in src_cell_val
               for filter_txt in self.disallowed_cells_text):
            return False

        # filter bad column names and
        if re.search('(?i)(N/A|[0-9]+px)|total|nbsp|px|/|\n|notes|total|-|,|\|',
                     table._table.header[source_column].column_name):
            return False

        if re.search('(?i)(N/A|[0-9]+px)|total|nbsp|px|/|\n|notes|total|-|,|\|',
                     table._table.header[target_column].column_name):
            return False

        # if no filter, keep fact
        return True

    def is_arithmetic_opp_possible(self, table, target_rows, header, answers, target_column, only_extractive):

        if len(answers) > 1 and header[target_column].metadata != {} and 'parsed_values' in header[
            target_column].metadata:
            # Filtering target inds
            # In some tables (such a sports statistics) the table itself contains a "Total" row which interfers with the
            # numeric reasoning. We filter these rows here, and don't consider them for the arithmetic operation.
            total_rows = [i for i, r in enumerate(table.table.table_rows) for j, c in enumerate(r) if
                          c.text.lower().find('total') > -1]
            target_rows = list(set(target_rows) - set(total_rows))

            if len(target_rows) == 0:
                return False, None, None, None
            else:
                target_column_header = header[target_column]
                parsed_values = pd.Series(target_column_header.metadata['parsed_values'])[target_rows].reset_index(
                    drop=True)

                # checking if the column is date:
                if re.search(r'year|years|date', target_column_header.column_name, re.IGNORECASE) is not None \
                        or ('DATE' in target_column_header.metadata['ner_appearances_map'] and \
                            target_column_header.metadata['ner_appearances_map']['DATE'] / \
                            len(target_column_header.metadata['parsed_values']) > 0.6):
                    is_date = True
                else:
                    is_date = False

                # randomly choosing operation (if it isn't given already)
                all_functions = {'max', 'min', 'mean', 'sum'}
                if is_date:
                    all_functions -= {'mean', 'sum'}
                if only_extractive:
                    all_functions -= {'mean', 'sum', 'count'}

                # we constraint min and max to have only one answer (no two rows with the min/max value)
                if (parsed_values == parsed_values.max()).sum() != 1:
                    all_functions -= {'max'}
                if (parsed_values == parsed_values.min()).sum() != 1:
                    all_functions -= {'min'}

                if len(all_functions) > 0:
                    opp = random.sample(list(all_functions), 1)[0]
                    return True, target_rows, opp, is_date
                else:
                    return False, None, None, None

        else:
            return False, None, None, None

    def filter_table_question(self, table, question):
        if self.filter_images_target_column(table, question):
            return True
        if self.filter_disallowed_source_cell(table, question):
            return True

        # filter target cells if the cell is an image or N/A
        for cell in question.metadata['table_cells_used_in_answer']:
            if re.search('(N/A|[0-9]+px)', table.table.table_rows[cell[0]][cell[1]].text):
                return True

        # filter source cells if the cell is an image or N/A
        for cell in question.metadata['table_cells_used_in_question']:
            if re.search('(N/A|[0-9]+px)', table.table.table_rows[cell[0]][cell[1]].text):
                return True

        return False

    def filter_images_target_column(self, table, question):
        target_column = question.metadata['target_column_id']
        images_column = {table.table.header[col].metadata['descriptive_image_column']
                         for col in range(table.n)
                         if 'descriptive_image_column' in table.table.header[col].metadata}
        if target_column in list(images_column):
            return True
        return False

    def filter_disallowed_source_cell(self, table, question):
        src_cell_ids = question.metadata['table_cells_used_in_question'][0]
        src_cell = table.table.table_rows[src_cell_ids[0]][src_cell_ids[1]]
        if any(filter_txt in src_cell.text.lower() for filter_txt in self.disallowed_cells_text):
            return True
        return False

    def add_arithmetic_opp(self, table, question_text, target_column_header, target_inds, template, opp, is_date,
                           target_column):

        parsed_values = pd.Series(target_column_header.metadata['parsed_values'])[target_inds].reset_index(drop=True)

        if opp == 'max':
            target_inds_selected = [target_inds[parsed_values.idxmax()]]
            if is_date:
                question_text = re.sub(template['arithmetic_max_date'][0], template['arithmetic_max_date'][1],
                                       question_text,
                                       flags=re.IGNORECASE)
            else:
                question_text = re.sub(template['arithmetic_max'][0], template['arithmetic_max'][1], question_text,
                                       flags=re.IGNORECASE)
            is_extractive = True
            # because this operation is extractive, we take the exact cell value as answer
            answers = [
                Answer(table.table.table_rows[target_inds_selected[0]][target_column].text, 'string', 'table', True, \
                       table_indices=[[t, target_column] for t in target_inds_selected])]
        elif opp == 'min':
            target_inds_selected = [target_inds[parsed_values.idxmin()]]
            if is_date:
                question_text = re.sub(template['arithmetic_min_date'][0], template['arithmetic_min_date'][1],
                                       question_text,
                                       flags=re.IGNORECASE)
            else:
                question_text = re.sub(template['arithmetic_min'][0], template['arithmetic_min'][1], question_text,
                                       flags=re.IGNORECASE)
            is_extractive = True
            # because this operation is extractive, we take the exact cell value as answer
            answers = [
                Answer(table.table.table_rows[target_inds_selected[0]][target_column].text, 'string', 'table', True, \
                       table_indices=[[t, target_column] for t in target_inds_selected])]
        elif opp == 'count':
            # in operations the target index will be all the cells use to computer the operation
            target_inds_selected = [target_inds[i] for i in range(len(parsed_values))]
            question_text = re.sub(template['arithmetic_count'][0], template['arithmetic_count'][1], question_text,
                                   flags=re.IGNORECASE)
            is_extractive = False
            answers = [Answer(int(parsed_values.count()), 'number', 'table', False, \
                              table_indices=[[t, target_column] for t in target_inds_selected])]
        elif opp == 'mean':
            # in operations the target index will be all the cells use to computer the operation
            target_inds_selected = [target_inds[i] for i in range(len(parsed_values))]
            question_text = re.sub(template['arithmetic_mean'][0], template['arithmetic_mean'][1], question_text,
                                   flags=re.IGNORECASE)
            is_extractive = False
            answers = [Answer(float(parsed_values.mean()), 'number', 'table', False, \
                              table_indices=[[t, target_column] for t in target_inds_selected])]
        elif opp == 'sum':
            # in operations the target index will be all the cells use to computer the operation
            target_inds_selected = [target_inds[i] for i in range(len(parsed_values))]
            question_text = re.sub(template['arithmetic_sum'][0], template['arithmetic_sum'][1], question_text,
                                   flags=re.IGNORECASE)
            is_extractive = False
            answers = [Answer(float(parsed_values.sum()), 'number', 'table', False, \
                              table_indices=[[t, target_column] for t in target_inds_selected])]

        # TODO K-th highest, comparative

        return question_text, target_inds_selected, answers, is_extractive

    def table_matches_template(self, table, template):
        match = True
        if 'page_title' in template and not re.search(template['page_title'], table.page_title, re.IGNORECASE):
            match = False
        if 'table_title' in template and not re.search(template['table_title'], table._table.table_name, re.IGNORECASE):
            match = False
        if 'not_table_title' in template and re.search(template['not_table_title'], table._table.table_name,
                                                       re.IGNORECASE):
            match = False
        if 'not_table_titles' in template:
            for table_title in template['not_table_titles']:
                if re.search(table_title, table._table.table_name, re.IGNORECASE):
                    match = False
        return match

    def config_check_val(self, prop, val):
        if prop in self._qgen_config:
            return self._qgen_config[prop] == val
        else:
            return False
