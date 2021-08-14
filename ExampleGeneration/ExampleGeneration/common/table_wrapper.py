import re


class WikiTable():
    def __init__(self, context):
        for d in context.context:
            if len(d.table.table_rows) > 0:
                self._table = d.table
                self.page_title = d.title
                self.page_url = d.url
                self.m = len(self.table.table_rows)
                self.n = len(self.table.table_rows[0])
                return
        else:
            assert(ValueError('no table in context!'))

    def get_cells_with_wiki_entities(self):
        links_set = set()
        for row in self._table.table_rows:
            for cell in row:
                for link in cell.links:
                    links_set.add(link)
        for link in links_set:
            yield link

    def get_wiki_entities_to_cells(self):
        url_to_wiki_entity_cells_pair = {}
        for row_idx, row in enumerate(self._table.table_rows):
            for col_idx, cell in enumerate(row):
                for link in cell.links:
                    if link.url not in url_to_wiki_entity_cells_pair:
                        url_to_wiki_entity_cells_pair[link.url] = (link, [])
                    url_to_wiki_entity_cells_pair[link.url][1].append((row_idx, col_idx))
        return {entity: cells for entity, cells in url_to_wiki_entity_cells_pair.values()}

    def generate_cells(self):
        for i, row in enumerate(self._table.table_rows):
            for j, cell in enumerate(row):
                yield cell

    def generate_columns(self):
        n_rows = len(self._table.table_rows)
        n_columns = len(self._table.table_rows[0])
        for column_ind in range(n_columns):
            column = [self._table.table_rows[i][column_ind] for i in range(n_rows)]
            yield column

    @staticmethod
    def get_link_permutations(link):
        """
        :param link:
        :return: a list with the two link's permutations in lower case
        """
        return [link.text.lower(), link.wiki_title.lower()]

    def is_link_column(self, column):
        col = []
        for row in self._table.table_rows:
            col.append(row[column])
        return sum([len(cell.links) for cell in col])

    @staticmethod
    def check_cell_equality(cell, content):
        """
        :param cell:
        :param content:
        :return: a boolean value if the cell is equal to the content
        """
        content = content.lower()
        cell_text = str(cell['text']).lower()
        if cell_text == content:
            return True
        elif len(cell['links']) == 1:
            cell_link_text = cell['links'][0]['text'].lower()
            if len(cell_link_text) + 5 >= len(cell_text):
                if cell_link_text == content:
                    return True
                if cell['links'][0]['wiki_title'].lower() == content:
                    return True
        return False

    def check_cell_in_list(self, cell, content_list):
        """
        :param cell:
        :param content_list:
        :return: return a boolean value. checks if at least one of the contents in the list is in the cell
        """
        for content in content_list:
            if self.check_cell_equality(cell, content):
                return True
        return False


    def get_answer_cells(self, answer):
        """
        :param table:
        :param answer:
        :return: get a list of cells in the table in which the answer appears
        """
        answer_cells = []
        for i, row in enumerate(self._table.table_rows):
            for j, cell in enumerate(row):
                if self.check_cell_equality(cell, answer):
                    answer_cells.append([i + 1, j])
        return answer_cells

    def get_answer_columns(self, answer):
        """
        :param table:
        :param answer:
        :return: a set of columns in which the answer appears
        """
        return {cell[1] for cell in self.get_answer_cells(self._table.table_rows, answer)}

    def get_entity_columns(self):
        return set([j for i, row in enumerate(self._table.table_rows) for j, cell in enumerate(row) if len(cell.links)])

    def get_row_inds(self):
        # We need to skip the header that is still part of the table.
        return set([i for i, row in enumerate(self._table.table_rows)])

    def get_column_inds(self):
        return set([j for i, row in enumerate(self._table.table_rows) for j, cell in enumerate(row)])

    def get_key_column(self):
        return set([i for i, column_descriptor in enumerate(self._table.header) if 'is_key_column' in column_descriptor.metadata])

    def get_entities_columns(self):
        return set([i for i, column_descriptor in enumerate(self._table.header) if 'entities_column' in column_descriptor.metadata])

    def get_arithmetic_inds(self):
        return set([i for i, column_descriptor in enumerate(self._table.header) if 'parsed_values' in column_descriptor.metadata])

    def match_column_inds_to_template(self, column_name_regex):
        return set([i for i, col in enumerate(self._table.header) if re.search(column_name_regex, col.column_name, re.IGNORECASE)])

    def row_cell_with_values(self, column_name_regex):
        return [i for i, r in enumerate(self._table.table_rows) \
                      for j, c in enumerate(r) if re.search(column_name_regex, c.text, re.IGNORECASE)]

    def get_rank_columns(self):
        return set([i for i, h in enumerate(self._table.header) if (
                    "parsed_values" in h.metadata and "type" not in h.metadata) or h.column_name.lower() in ['rank', 'week']])

    def get_link_in_cell(self, row, column):
        if len(self._table.table_rows[row][column].links) == 1:
            return [self._table.table_rows[row][column].links[0].to_json()]
        return []

    @staticmethod
    def all_cells_from_same_column(cells):
        cells_size = len(cells)
        if cells_size:
            first_col = cells[0][1]
            return cells_size == len([cell for cell in cells if cell[1] == first_col])
        return False

    @property
    def table(self):
        return self._table
