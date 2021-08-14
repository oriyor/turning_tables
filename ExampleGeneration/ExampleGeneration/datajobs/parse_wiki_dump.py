import logging
from tqdm import tqdm

from ExampleGeneration.common.wikipedia_dump_utils import iterate_articles
from ExampleGeneration.datajob import DataJob
import ExampleGeneration.wikitextparser as wtp
from ExampleGeneration.common.multiqa_format_wrapper import *
import os, gzip
from ExampleGeneration.common.file_utils import CACHE_DIRECTORY, cached_path
from ExampleGeneration.common.file_utils import upload_local_file_to_s3
from ExampleGeneration.common.multi_process_streaming import split
import multiprocessing
import re
logger = logging.getLogger(__name__) # pylint: disable=invalid-name


class ParseWikiDumpDataJob(DataJob):

    def __init__(self, datajob_name, args):

        self.datajob_name = datajob_name
        logger.info("loading...")
        super().__init__(args)

        self.input_path = self._config["dump_file_path"]

        self.min_table_rows = self._config["min_table_rows"]
        self.min_table_cols = self._config["min_table_cols"]
        self.max_table_rows = self._config["max_table_rows"]

        self.two_rows_header_cnt = 0
        self.three_rows_header_cnt = 0
        self.tables_count = 0
        self.tables_with_dim_cnt = 0

    def run_datajob(self, args=None):
        if ".bz2" in self.input_path:
            self.parse_tables_single_dump_file(self.input_path, self.output_path)
        else:
            logger.info(f"multiprocess parsing all files in input dir")
            n_processes = self._config["n_processes"]
            dump_files_chunks = split(os.listdir(self.input_path), n_processes)
            pool = multiprocessing.Pool(processes=n_processes)
            pool.map(self.parse_chunk_of_dump_files, dump_files_chunks)
            logger.info(f"done parsing all files")

    def parse_chunk_of_dump_files(self, input_f_name_chunk):
        for input_f_name in input_f_name_chunk:
            input_path = os.path.join(self.input_path, input_f_name)
            output_path = os.path.join(os.path.dirname(self.output_path), input_f_name.replace('.bz2', '_parsed.gz'))
            self.parse_tables_single_dump_file(input_path, output_path)

    def parse_tables_single_dump_file(self, input_path, output_path):
        logger.info(f"start paring {input_path}")

        # We assume that if output is not meant to be save in s3, then this is a test sample
        # meant for debugging.
        if 's3' in output_path:
            s3_output_file = output_path.replace('s3://', '')
            cache_output_path = os.path.join(CACHE_DIRECTORY, s3_output_file.replace('/', '_'))
            if ".gz" in s3_output_file:
                output_fp = gzip.open(cache_output_path, 'wb')
            else:
                output_fp = open(cache_output_path, 'w')
        else:
            if ".gz" in output_path:
                output_fp = gzip.open(output_path, 'wb')
            else:
                output_fp = open(output_path, 'w', encoding='utf-8')

        pbar = tqdm()
        for i, (title, page_id, raw_text) in enumerate(iterate_articles(cached_path(input_path))):
            if i == self._config["max_number_of_examples"]:
                break

            parsed = self.parse_wiki_tables_from_dump_page(title, page_id, raw_text)
            for c in parsed:
                if 's3' in output_path:
                    output_fp.write((str(c) + '\n').encode('utf-8'))
                else:
                    # We assume that if output is not meant to be save in s3, then this is a test sample
                    # meant for debugging.
                    #self.write_table_to_file(c, output_fp)
                    output_fp.write((str(c) + '\n').encode('utf-8'))
            pbar.update(1)

        logging.info(f"n raw tables: {self.tables_count}")
        logging.info(f"tables within dim freq: {self.tables_with_dim_cnt/float(self.tables_count)} ({self.tables_with_dim_cnt})")
        logging.info(f"2 rows header freq: {self.two_rows_header_cnt/float(self.tables_with_dim_cnt)}")
        logging.info(f"3 rows header freq: {self.three_rows_header_cnt/float(self.tables_with_dim_cnt)}")
        pbar.close()
        output_fp.close()
        if 's3' in output_path:
            upload_local_file_to_s3(cache_output_path, s3_output_file)

    @staticmethod
    def get_cell_entity(parsed_cell):
        txt = ParseWikiDumpDataJob.handle_templates(parsed_cell)
        txt = ParseWikiDumpDataJob.remove_html_tags(txt)
        txt = txt.replace('\'', '')  # in wiki text '' mean bold/italics
        links = []
        for lnk in parsed_cell.wikilinks:
            lnk_text = lnk.target if lnk.text is None else lnk.text
            txt = txt.replace(lnk.string, lnk_text)
            links.append(WikiLink(lnk_text, lnk.target))
        return TableCell(txt, links)

    @staticmethod
    def write_table_to_file(context, output_fp):
        """
        :param context:
        :param output_fp:
        :return: outputs the contexts table to output_fp, in the following order:
                    1. table url
                    2. column names
                    3. rows
        """
        output_fp.write(context.context[0].url + '\n')
        wiki_table = context.context[0].table
        line_str = ' | '.join([column_descriptor.column_name for column_descriptor in wiki_table.header])
        output_fp.write(line_str + '\n')
        for t_line in wiki_table.table_rows:
            line_str = ' | '.join([cell.text for cell in t_line])
            output_fp.write(line_str + '\n')
        output_fp.write('\n')

    @staticmethod
    def handle_templates(parsed_cell):
        """
        Wiki text can have templates (see https://en.wikipedia.org/wiki/Help:Template)
        :param parsed_cell: a parsed cell object (parsed by wtp)
        :return:
        the text of the cell with the following changes
        1.  without external link templates
            for example: {{cite encyclopedia| last = french| first = e.l| encyclopedia = australian dictionary of biography| title = morrison, alexander (1829 - 1903)| url = http://www.adb.online.anu.edu.au/biogs/a050341b.htm?hilite=scotch%3bcollege| accessdate = 2008-03-26| edition = online| year   =1974| publisher = melbourne university press| volume = 5| location = melbourne, vic. | pages=295–297}}
        2. without 'flagicon' template (tells the html parser to put an icon of a flag)
        3. with the tooltip template formatted as "phrase (acronym)"
            for example: {{tooltip|3p%|3 point field goal percentage}} -> 3 point field goal percentage (3p%)
        """
        txt = parsed_cell.string
        for template in parsed_cell.templates:
            template_name = template.name.lower()
            if any([t == template_name for t in ["sort", "sortname"]]):
                value = template.arguments[-1].value
                txt = txt.replace(template.string, value)
            # the tooltips usually looks like {Tooltip|$acronym|$phrase} i.e phrase=template.arguments[-1] etc
            elif "tooltip" in template_name:
                phrase = template.arguments[-1].value
                acronym = "" if len(template.arguments) < 2 else f" ({template.arguments[-2].value})"
                txt = txt.replace(template.string, f"{phrase}{acronym}")
            else:
                txt = txt.replace(template.string, "")
        return txt

    def get_table_entity(self, rows):
        parsed_table = [[None] * len(r) for r in rows]
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if cell is None:
                    parsed_table[i][j] = TableCell(text="", links=[])
                else:
                    parsed_table[i][j] = self.get_cell_entity(wtp.parse(cell))
        return parsed_table

    def add_table_header(self, parsed_table):
        """
        :param parsed_table:
        :return: returns the parsed table with a valid header, which is a list of WikiTableColumnDescriptors
        """
        parsed_table = self.fix_table_titles_as_header(parsed_table)
        parsed_table = self.handle_multi_row_header(parsed_table)
        if parsed_table is not None:
            parsed_table.header = [WikiTableColumnDescriptor(column_header.text) for column_header in parsed_table.table_rows[0]]
            del parsed_table.table_rows[0]
        return parsed_table

    @staticmethod
    def fix_table_titles_as_header(parsed_table):
        """
        In some cases the table title is in the table header
        example: https://en.wikipedia.org/wiki/Goodenough_College
        in this case, all the column titles will be the same
        so we would like to use the first row as table title and the second as header
        """
        header = parsed_table.table_rows[0]
        if len(header) > 1:
            n_unique_column_titles = len(set([t.text for t in header]))
            if n_unique_column_titles == 1:
                #todo this is the table_title
                parsed_table.table_name = f"{parsed_table.table_name} | {header[0].text}"
                del parsed_table.table_rows[0]  # removing header from table
        return parsed_table

    @staticmethod
    def get_wiki_url_from_title(title):
        return "https://en.wikipedia.org/wiki/{}".format(title.replace(' ', '_'))

    def parse_wiki_tables_from_dump_page(self, title, page_id, raw_text):
        result = []
        parsed = wtp.parse(raw_text)
        #todo this is where we parse the wiki to wikititle
        for table_i, table in enumerate(parsed.tables):
            self.tables_count += 1
            table_id = f"{page_id}-{table_i}"
            rows = table.data(span=True)

            if not (self.min_table_rows <= len(rows) <= self.max_table_rows)\
                    or len(rows[0]) < self.min_table_cols:  # Filter tables by dimension
                continue

            self.tables_with_dim_cnt += 1
            parsed_table = WikiTable(table_rows=self.get_table_entity(rows), table_name=table.table_title)
            parsed_table = self.add_table_header(parsed_table)
            if parsed_table is not None:
                doc = Doc(title=title, table=parsed_table, url=self.get_wiki_url_from_title(title),
                          caption=table.table_title,
                          metadata={"id": table_id, "type": "wiki-table"})

                multi_modal = MultiQaModel(id=table_id, context=[doc], qas=[])
                result.append(multi_modal)
        return result

    def handle_multi_row_header(self, parsed_table):
        """
        if header has one level - do nothing

        if header has two levels - join them to a single row header
        example: https://en.wikipedia.org/wiki/Mexico_at_the_1984_Winter_Olympics

        if header has three levels - return None to indicate that we would like to
        throw away this context (because the header is too complex)
        example: https://en.wikipedia.org/wiki/Brazil_at_the_1948_Summer_Olympics (Athletics)

        :param doc: a Doc object of a wiki table
        """
        header = [t.text for t in parsed_table.table_rows[0]]

        # If there are 2 following columns with the same name we will address it as a multi rows header
        is_multi_row_header = any([(prev == cur and prev and cur) for prev, cur
                                  in zip(header, header[1:])])

        if is_multi_row_header and len(parsed_table.table_rows) > 1:
            sub_header = [t.text for t in parsed_table.table_rows[1]]

            if any([(prev == cur and prev and cur) for prev, cur
                    in zip(sub_header, sub_header[1:])]):
                self.three_rows_header_cnt += 1
                return None  # has three levels header

            self.two_rows_header_cnt += 1
            parsed_table = self.join_multi_row_header(parsed_table)  # has two levels header

        return parsed_table

    @staticmethod
    def join_multi_row_header(parsed_table):
        """
        :param doc: Doc object of a wiki table with two levels header
                    example: https://en.wikipedia.org/wiki/Mexico_at_the_1984_Winter_Olympics
        :return: Doc object with the first two rows joined as header
        """
        header = [t.text for t in parsed_table.table_rows[0]]
        sub_header = [t.text for t in parsed_table.table_rows[1]]  # the header to join to the first row
        # sometimes there is row span in the header, in this case we don't want to join col titles
        # example: https://en.wikipedia.org/wiki/Bulgaria_at_the_1988_Winter_Olympics (Event column)
        joined_header = [TableCell(text=f"{h} - {s}" if h != s else h, links=[])
                         for h, s in zip(header, sub_header)]

        # delete original two first rows and replace them with the joined header
        del parsed_table.table_rows[:2]
        parsed_table.table_rows.insert(0, joined_header)

        return parsed_table

    @staticmethod
    def remove_html_tags(string):
        pattern = re.compile(r'<.*?>')
        result = pattern.sub(' ', string)
        return result
