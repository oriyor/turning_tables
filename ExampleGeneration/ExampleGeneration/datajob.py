import json
import logging
import os
from ExampleGeneration.common.file_utils import upload_jsonl_to_s3, save_jsonl_to_local, is_path_creatable
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataJob():

    def __init__(self, args, load_context_as_multiqa=False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations",\
                                                   args.config_file_name)
        with open(config_path, 'r') as f:
            self._config = json.load(f)[self.datajob_name]

            if "input_file" in args and args.input_file is not None:
                self.input_path = self.get_path(args.working_directory, args.input_file)
            elif "input_file" in self._config:
                self.input_path = self.get_path(args.working_directory, self._config["input_file"])

            if "output_file" in args and args.output_file is not None:
                self.output_path = self.get_path(args.working_directory, args.output_file)
            elif "output_file" in self._config:
                self.output_path = self.get_path(args.working_directory, self._config["output_file"])

    @staticmethod
    def get_path(dir, file_name):
        if "/" in file_name:  # if full path simply use it
            path = file_name
        else:
            assert len(dir) > 0, "No directory has been specified"
            path = os.path.join(dir, file_name)
        return path

    def save_output(self):
        if self._config["output_file"].startswith('s3://'):
            save_func = upload_jsonl_to_s3
        elif is_path_creatable(self._config["output_file"]) and len(self._config["output_file"]) > 0:
            save_func = save_jsonl_to_local
        else:
            # Do nothing
            return

        save_func(self._config["output_file"], self.datajob_output['contexts'], self.datajob_output.get('header', None))

    def run_datajob(self,args):
        pass
