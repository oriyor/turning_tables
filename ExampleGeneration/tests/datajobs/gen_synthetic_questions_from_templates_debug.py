import pytest
import argparse
import os, json

from ExampleGeneration.common.analysis_utils import dump_manual_analysis_facts
from ExampleGeneration.datajobs.gen_synthetic_questions_from_templates import \
    GenSyntheticQuestionsFromTemplatesDataJob


class TestSyntheticGenQuestionsFromTemplatesDEBUG:

    def test_synthetic_reasoning_questions(self):
        config_file = "config_reas.json"
        config_entry = "GenQuestionsFromTemplates_TabReas"
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

        # loading the question template externally, to control which questions to produce:
        # with open(os.path.join('ExampleGeneration', 'question_templates', question_templates_file)) as f:
        #     q_templates = process_question_templates(json.load(f))
        # curr_question_gen_template = [t for t in q_templates if t['name'] == question_generator_name]
        # curr_question_gen_template[0]['enable'] = True

        datajob = GenSyntheticQuestionsFromTemplatesDataJob(config_entry, args)

        # reducing data size to a sample:
        datajob._config['n_processes'] = 1
        datajob._config['max_chunk_size'] = 100
        datajob._config['max_number_of_examples'] = 100
        datajob.input_path = "data/datajob_samples/classify_table_column_types_sample.jsonl"
        datajob.output_path = "data/datajob_samples/synthetic_questions.jsonl"

        datajob.run_datajob(args)

        dump_manual_analysis_facts('data/datajob_samples/synthetic_questions.jsonl', \
                                   'data/datajob_samples/synthetic_questions.csv')
