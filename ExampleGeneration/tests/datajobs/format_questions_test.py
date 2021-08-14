import pytest
import argparse
import os, json

from ExampleGeneration.datajobs.format_questions import FormatQuestionsDataJob
from ExampleGeneration.common.analysis_utils import dump_synthetic_questions_analysis


class TestFormatQuestionsDEBUG:

    def test_format_questoins(self):
        config_file = "config_reas.json"
        config_entry = "FormatSyntheticQuestions"
        working_directory = "data/tab_reas"

        parse = argparse.ArgumentParser("")
        parse.add_argument("-c", "--datajob_name", type=str, help="The name of the datajob class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("-config", "--config_file_name", type=str, help="", default="config_reas.json")
        parse.add_argument("-wd", "--working_directory", type=str, help="dir of input file, can be s3 path")
        parse.add_argument("-af", "--annotated_questions_file", type=str, help="dir of input file, can be s3 path",
                           default=None)

        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(
            ["-c", "Example", "-o", "build_datajob", "-config", config_file, "-wd", working_directory])

        datajob = FormatQuestionsDataJob(config_entry, args)

        # reducing data size to a sample:
        datajob._config['n_processes'] = 1
        datajob._config['max_chunk_size'] = 1000
        datajob._config['max_number_of_examples'] = 1000
        datajob.input_path = "data/datajob_samples/synthetic_questions.jsonl"
        datajob.output_path = "data/datajob_samples/formatted_synthetic_questions.jsonl"
        datajob.run_datajob(args)

        dump_synthetic_questions_analysis('data/datajob_samples/formatted_synthetic_questions.jsonl', \
                                          'data/datajob_samples/formatted_synthetic_questions.csv')
