# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
from ExampleGeneration.datajobs.reas_classify_column_types import ReasClassifyColumnTypesDataJob


class TestClassifyTableColumnTypesDataJob:
    def test_run_datajob(self):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-c", "--datajob_name", type=str, help="The name of the datajob class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("-config", "--config_file_name", type=str, help="DataJobs config file name", default="config_reas.json")
        parse.add_argument("-wd", "--working_directory", type=str, help="dir of input file, can be s3 path")
        parse.add_argument("-af", "--annotated_questions_file", type=str, help="dir of input file, can be s3 path", default=None)
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-c", "AddColumnTypeMetadata","-o", "build_datajob",
                                 "-wd", "data"])

        datajob = ReasClassifyColumnTypesDataJob('ReasClassifyColumnTypes',args)

        # reducing data size to a sample:
        datajob._config['max_number_of_examples'] = 100
        datajob._config['max_number_of_add_column_type_metadatas'] = 100
        datajob.output_path = "data/datajob_samples/classify_table_column_types_sample.jsonl"

        datajob.run_datajob(args)
