# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
from ExampleGeneration.datajobs.parse_wiki_dump import ParseWikiDumpDataJob
import os

class TestParseWikiDump:
    def test_run_datajob(self):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-c", "--datajob_name", type=str, help="The name of the datajob class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("-config", "--config_file_name", type=str, help="DataJobs config file name", default="config_tests.json")
        parse.add_argument("-wd", "--working_directory", type=str, help="dir of input file, can be s3 path")

        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-c", "Example","-o", "build_datajob",
                                 "-out","parse_wiki_dump_sample.jsonl",
                                 "-wd", "data"])

        datajob = ParseWikiDumpDataJob('ParseWikiDump',args)

        # reducing data size to a sample:
        datajob._config['max_number_of_examples'] = 100

        # Seems this is the default in the config anyway ..
        datajob.output_path = os.path.join("data", "datajob_samples",
                                           "parse_wiki_dump_sample.gz")

        datajob.run_datajob(args)
