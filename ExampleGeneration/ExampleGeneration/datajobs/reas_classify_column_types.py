import json
import logging

from tqdm import tqdm
from dateutil import parser
from datetime import datetime
import spacy
import numpy as np

from ExampleGeneration.common.multi_process_streaming import multi_process_data_stream
from ExampleGeneration.common.multiqa_format_wrapper import MultiQaModel
from ExampleGeneration.common.table_wrapper import WikiTable
from ExampleGeneration.datajob import DataJob

nlp = spacy.load("en_core_web_sm")
today = datetime.today()
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_images_suffix():
    return np.array([".jpg", ".png", ".tiff", ".jpeg", ".bmp"])


image_suffixes = get_images_suffix()


class ReasClassifyColumnTypesDataJob(DataJob):
    def __init__(self, datajob_name, args):
        self.datajob_name = datajob_name
        logger.info("loading...")
        super().__init__(args)

    @staticmethod
    def contains_word(s, w):
        return f' {w} ' in f' {s} '

    @staticmethod
    def add_ner_labels_metadata(table_wrapper):
        """
        :param table_wrapper:
        :return: adds a map to the table's metadata, that contains the number of appearances for each
                    named entity appearance. The labels are found using spacy ner
        """
        for col_ind in range(table_wrapper.n):
            col_ner_label_appearances_map = {}
            column_values = [table_wrapper.table.table_rows[i][col_ind].text for i in range(table_wrapper.m)]
            column_text_filtered_values = [val for val in column_values if type(val) == str]

            for cell_text in column_text_filtered_values:
                doc = nlp(cell_text)
                if len(doc.ents) == 1:
                    label = doc.ents[0].label_
                    if label not in col_ner_label_appearances_map:
                        col_ner_label_appearances_map[label] = 0
                    col_ner_label_appearances_map[label] = col_ner_label_appearances_map[label] + 1

            table_wrapper.table.header[col_ind].metadata["ner_appearances_map"] = col_ner_label_appearances_map

    @staticmethod
    def add_time_and_int_column_types(table_wrapper):
        """
        :param table_wrapper:
        :return: adds the following data to the table's metadata:
                    1. The index column if one exists, otherwise None
                    2.  For every column that can be fully parsed to numeric values,
                        a map between the column id and the column parsed to floats,
                        such that it could be used for sorting
                    3. For every column that can be fully parsed to datetime values,
                        a map between the column id and the column parsed to datetime,
                        such that it could be used for sorting
        """
        for col_ind, column in enumerate(table_wrapper.generate_columns()):
            date_validity_flag, datetime_column = ReasClassifyColumnTypesDataJob.try_parse_column_to_datetime_column(
                column)
            if date_validity_flag:
                table_wrapper.table.header[col_ind].metadata["type"] = "Datetime"
                table_wrapper.table.header[col_ind].metadata["parsed_values"] = datetime_column
            int_validity_flag, int_column = ReasClassifyColumnTypesDataJob.try_parse_column_to_int_column(column)
            if not date_validity_flag and int_validity_flag:
                table_wrapper.table.header[col_ind].metadata["parsed_values"] = int_column
                table_wrapper.table.header[col_ind].metadata["sorted"] =  ReasClassifyColumnTypesDataJob.is_sorted_column(int_column)
                if not ReasClassifyColumnTypesDataJob.is_index_column(int_column):
                    table_wrapper.table.header[col_ind].metadata["type"] = "float"
                else:
                    table_wrapper.table.header[col_ind].metadata["is_index_column"] = True

    @staticmethod
    def try_parse_column_to_int_column(column):
        """
        :param column:
        :return: a tuple:
                    1. a validity flag
                    2. if successful - the column parsed to ints
        """
        int_col = []
        try:
            for cell in column:
                int_col.append(float(cell.text.replace(',', '')))
            return True, int_col
        except ValueError:
            return False, []

    @staticmethod
    def is_sorted_column(column):
        """
        :param column:
        :return: a boolean value to determine if this is a static column
        """
        # check if the column is sorted up
        res = True
        for i in range(1, len(column)):
            if column[i] <= column[i - 1]:
                res = False
        if res:
            return True
        # check if the column is sorted down
        res = True
        for i in range(1, len(column)):
            if column[i] >= column[i - 1]:
                res = False
        if res:
            return True
        return False

    @staticmethod
    def is_index_column(column):
        """
        :param column:
        :return: a boolean value to determine if this is a static column
        """
        for i in range(1, len(column)):
            if column[i] - column[i - 1] != 1:
                return False
        return True

    @staticmethod
    def try_parse_column_to_datetime_column(column):
        """
        :param column:
        :return: a tuple:
                    1. a validity flag
                    2. if successful - the column parsed to datetime
        """
        datetime_col = []
        try:
            for cell in column:
                # TODO this is a workeraround for an error (see Trello)
                    parsed_time = parser.parse(cell.text)
                    # todo discuss if this is a good assumption
                    if parsed_time.year < today.year\
                            and (parsed_time.month != today.month or parsed_time.day != today.day):
                        datetime_col.append(parsed_time.isoformat())
                    else:
                        return False, []

            return True, datetime_col
        except ValueError:
            return False, []
        except OverflowError:
            return False, []
        except TypeError:
            return False, []

    @staticmethod
    def add_key_column(table_wrapper):
        """
        :param table_wrapper:
        :return: adds a key column to the table's metadata, but only on success.
                    We scan the table's columns from the left, until finding a distinct column which
                    is not the index column
        """
        m = table_wrapper.m
        n = table_wrapper.n
        for col_ind in range(n):
            column_text = [table_wrapper.table.table_rows[i][col_ind].text for i in range(m)]
            if len(set(column_text)) == m:
                if "is_index_column" not in table_wrapper.table.header[col_ind].metadata:
                    table_wrapper.table.header[col_ind].metadata["is_key_column"] = True
                    break

        key_column_features = {}
        key_column_features_weighted = []
        entity_column_features_weighted = []

        for col_ind in range(n):
            distance_from_left = 1-(col_ind/n)
            percentage_with_entities = len([1 for i in range(m) if len(table_wrapper.table.table_rows[i][col_ind].links) >= 1])/m
            percentage_with_large_text = len([1 for i in range(m) if len(table_wrapper.table.table_rows[i][col_ind].text) >= 3])/m

            column_text = [table_wrapper.table.table_rows[i][col_ind].text for i in range(m)]
            uniqueness = len(set(column_text))/m

            numeric = 0 if "type" not in table_wrapper.table.header[col_ind].metadata else 1
            not_cardinals = 1
            if 'CARDINAL' in table_wrapper.table.header[col_ind].metadata['ner_appearances_map']:
                cardinal_appearnaces = table_wrapper.table.header[col_ind].metadata['ner_appearances_map']['CARDINAL']
                not_cardinals = 1-(cardinal_appearnaces/m)

            name_title = 0
            if ReasClassifyColumnTypesDataJob.contains_word(table_wrapper.table.header[col_ind].column_name.lower(), 'name')\
                    or ReasClassifyColumnTypesDataJob.contains_word(table_wrapper.table.header[col_ind].column_name.lower(), 'title'):
                name_title = 1

            image_cells = [1 for i in range(m) if len(table_wrapper.table.table_rows[i][col_ind].links) == 1 and
                           any(suff in table_wrapper.table.table_rows[i][col_ind].links[0].wiki_title.lower() for suff in image_suffixes)]
            non_image_ratio = 1 - (len(image_cells)/m)

            key_column_features[col_ind] = {'distance_from_left': distance_from_left,
                                            "entities": percentage_with_entities,
                                            "uniqueness": uniqueness,
                                            "not_numbers_ner": not_cardinals,
                                            "large_text": percentage_with_large_text,
                                            "not_images": non_image_ratio,
                                            "numeric_or_date": numeric*-10,
                                            "name_title": name_title*5}

            key_column_features_weighted.append(np.average([distance_from_left, percentage_with_entities, uniqueness, not_cardinals, percentage_with_large_text, non_image_ratio
                                                         ]) + name_title*5 + numeric*-10)
            entity_column_features_weighted.append(np.average([percentage_with_entities, uniqueness, not_cardinals, percentage_with_large_text, non_image_ratio]))


        #add image associated column for in line images
        key_column = np.argmax(key_column_features_weighted)
        table_wrapper.table.header[key_column].metadata["image_associated_column"] = True

        #add entities column flag
        for col in list(np.arange(n)[np.argwhere(np.array(entity_column_features_weighted) >= 0.9)].reshape(-1)):
            table_wrapper.table.header[col].metadata["entities_column"] = True

    @staticmethod
    def add_wiki_entity_links_for_column(table_wrapper):
        """
        :param table_wrapper:
        :return: adds a mapping with the number of wiki_entity appearances for each column
        """
        wiki_entity_links_for_column = {column: table_wrapper.is_link_column(column) for column in
                                        range(len(table_wrapper.table.table_rows[0]))}
        for col_ind, num_of_links in wiki_entity_links_for_column.items():
            table_wrapper.table.header[col_ind].metadata["num_of_links"] = num_of_links

    @staticmethod
    def add_entity_appearances_metadata(table_wrapper):
        """
        :param table_wrapper:
        :return: adds map between wiki entities and their appearances in the table
        """
        #todo add the image_s3_url in image datajob
        entities = {cell.links[0].wiki_title: {"text": cell.links[0].text, "url": cell.links[0].url, "table_cells": [],
                                    "image_s3_url": ""} for i, row in enumerate(table_wrapper._table.table_rows) for
                                    j, cell in enumerate(row) if len(cell.links) == 1}
        for i,row in enumerate(table_wrapper._table.table_rows):
            for j, cell in enumerate(row):
                if len(cell.links) == 1:
                    entities[cell.links[0].wiki_title][ "table_cells"].append([i,j])
        table_wrapper.table.metadata['entities_appearances'] = entities

    @staticmethod
    def add_image_columns(table_wrapper):
        """
        :param table_wrapper:
        :return: adds a mapping between entities column and images column using descriptive_image_column flag
        """
        #todo add the image_s3_url in image datajob
        table = np.array(table_wrapper.table.table_rows)
        for i, col in enumerate(table.T):
            n_real_images = 0

            for cell in col:
                # If there is exactly one image file in the
                # cell
                if len(cell.links) == 1 and any(suff in cell.links[0].wiki_title.lower() for suff in image_suffixes):
                    n_real_images += 1

            if n_real_images >= 1:
                # find source column
                key_columns = [k for k, x in enumerate(table_wrapper.table.header) if 'image_associated_column' in x.metadata]
                if len(key_columns):
                    key_column = key_columns[0]
                    table_wrapper.table.header[key_column].metadata['descriptive_image_column'] = i

                break

    def run_datajob(self, args):
        multi_process_data_stream(self.input_path, self.output_path,
                                  apply_on_lines_chunk=self.classify_column_types,
                                  n_processes=self._config["n_processes"],
                                  max_chunk_size=self._config["max_chunk_size"],
                                  max_lines_to_process=self._config["max_number_of_examples"], args=args)

    def classify_column_types(self, contexts):
        contexts = [MultiQaModel.from_json(json.loads(c)) for c in contexts]
        for context in tqdm(contexts):

            table_wrapper = WikiTable(context)
            self.add_time_and_int_column_types(table_wrapper)
            self.add_wiki_entity_links_for_column(table_wrapper)
            self.add_ner_labels_metadata(table_wrapper)
            self.add_entity_appearances_metadata(table_wrapper)
            self.add_key_column(table_wrapper)
            self.add_image_columns(table_wrapper)

        return contexts
