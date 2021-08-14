import argparse

from ExampleGeneration.datajob_factory import DataJobFactory

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-c", "--datajob_name", type=str, help="The name of the datajob class and config to use")
    parse.add_argument("-o", "--operation", type=str, help="The task stage to run", default='run_datajob')
    parse.add_argument("-in", "--input_file", type=str, help="")
    parse.add_argument("-out", "--output_file", type=str, help="")
    parse.add_argument("-config", "--config_file_name", type=str, help="DataJobs config file name", default="config_tests.json")
    parse.add_argument("--copy_from", type=str, help="For create new challenge, the chllenge to copy from", default=-1)
    parse.add_argument("--datajob_module", type=str, help="For create new challenge, the target challenge path", default='')
    parse.add_argument("-wd", "--working_directory", type=str, help="dir of input file, can be s3 path", default='')
    parse.add_argument("-af", "--annotated_questions_file", type=str, help="dir of input file, can be s3 path", default=None)

    # In the test no output file will be produced, change -out to create an output
    args = parse.parse_args()


    if args.operation == 'create_new_datajob':
        DataJobFactory().create_new_datajob(args.datajob_name, args.datajob_name, args)
    else:
        datajob = DataJobFactory().get_datajob(args.datajob_name, args.datajob_name, args)
        if args.operation == 'run_datajob':
            datajob.run_datajob(args)
        else:
            logger.error('Operation not supported')

if __name__ == '__main__':
    main()
