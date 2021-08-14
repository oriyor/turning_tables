import argparse
import gzip
import json
import os
import logging
import platform
from ExampleGeneration.datajob_factory import DataJobFactory

from ExampleGeneration.common.analysis_utils import dump_synthetic_questions_analysis
from ExampleGeneration.common.file_utils import cached_path, upload_local_file_to_s3

logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-c", "--datajob_name", type=str, help="The name of the datajob class and config to use")
    parse.add_argument("-config", "--config_file_name", type=str, help="DataJobs config file name", default="config_tests.json")
    parse.add_argument("-wd", "--working_directory", type=str, help="dir of input file, can be s3 path", default='')
    parse.add_argument("-sc", "--start_chunk", type=int, help="dir of input file, can be s3 path", default=0)
    parse.add_argument("-ec", "--end_chunk", type=int, help="dir of input file, can be s3 path", default=None)
    parse.add_argument("-af", "--annotated_questions_file", type=str, help="dir of input file, can be s3 path", default=None)
    parse.add_argument("-mf", "--filename_to_merge", type=str, help="dir of input file, can be s3 path", default=None)
    parse.add_argument("-dj", "--datajobs_to_run", type=str, help="A list of datajobs (it will check if each is enabled)", default=None)
    parse.add_argument("--build_train_dev_sets", action='store_true', help="upload dev train splits", default=False)

    # In the test no output file will be produced, change -out to create an output
    args = parse.parse_args()


    if args.datajobs_to_run is not None:
        args.datajobs_to_run = args.datajobs_to_run.split(',')
    base_working_directory = args.working_directory

    args.base_working_directory = base_working_directory
    if args.annotated_questions_file is not None:
        args.annotated_questions_file = base_working_directory + args.annotated_questions_file

    for chunk_num in range(args.start_chunk,args.end_chunk + 1):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations", \
                                   args.config_file_name)
        with open(config_path, 'r') as f:
            _config = json.load(f)

        # TODO better this be configurable:
        logger.info(f"----------- Working on chunk {'{:04d}'.format(chunk_num)} ------------- ")

        args.working_directory = base_working_directory + 'chunk_' + '{:04d}'.format(chunk_num) + '/'

        # loop over all the datajobs
        for datajob_name, datajob_config in _config.items():
            if (args.datajobs_to_run is not None and datajob_name in args.datajobs_to_run) or \
                    (args.datajobs_to_run is None and datajob_config['enable']):
                if datajob_name == 'FilterWikiTables':
                    args.input_file = _config['FilterWikiTables']["input_file"].replace('chunk_0000','chunk_' + '{:04d}'.format(chunk_num))
                if datajob_name == 'ReasClassifyColumnTypes':
                    args.input_file = _config['ReasClassifyColumnTypes']["input_file"].replace('chunk_0000','chunk_' + '{:04d}'.format(chunk_num))
                elif 'input_file' in args:
                    del args.input_file

                logger.info("-------------- Running: " + datajob_name + " --------------------")
                datajob = DataJobFactory().get_datajob(datajob_name, datajob_config['type'], args)
                datajob.run_datajob(args)

    # Append all dataset to the full dataset.
    if args.filename_to_merge is not None:
        for filename_to_merge in args.filename_to_merge.split(','):

            logger.info(f'\n-------Merging file {filename_to_merge}-----------\n')

            questions = []
            for chunk_num in range(args.start_chunk, args.end_chunk + 1):
                chunk_dataset_path = cached_path(base_working_directory + 'chunk_' + '{:04d}'.format(chunk_num) + '/' + filename_to_merge)
                with gzip.open(chunk_dataset_path, 'r') as f:
                    #header = f.readline()
                    for line in f:
                        question = json.loads(line)
                        questions.append(question)

            with gzip.open(filename_to_merge, 'w') as f:
                for line in questions:
                    f.write((json.dumps(line) + '\n').encode('utf-8'))

            upload_local_file_to_s3(filename_to_merge, base_working_directory.replace('s3://', '') + filename_to_merge)

            # dump csv if local
            local_platform = platform.node() == 'Oris-MacBook-Pro.local'

            logger.info(f'---- Calculating stats, local platfrom {local_platform} ----')

            dump_synthetic_questions_analysis(filename_to_merge, \
                                              'data/tab_reas/samples/template_q_sample_full_wip.csv',
                                              dump_csv=local_platform)

            # remove local if not local platform
            if not local_platform:
                os.remove(filename_to_merge)

            print()

if __name__ == '__main__':
    main()