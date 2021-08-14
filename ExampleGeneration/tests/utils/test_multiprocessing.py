# pylint: disable=no-self-use,invalid-name
import pytest, os
from ExampleGeneration.common.multi_process_streaming import multi_process_lst, multi_process_data_stream
from ExampleGeneration.common.file_utils import CACHE_DIRECTORY

class TestMultiProcessing:

    @staticmethod
    def apply_on_chunk(chunk):
        return [item*3 for item in chunk]

    @staticmethod
    def apply_on_lines(lines):
        return [int(line)*3 for line in lines]

    def test_multi_process_lst(self):
        lst = [1]*100000
        res = multi_process_lst(lst, self.apply_on_chunk, chunk_size=1000, n_processes=5)
        assert res == self.apply_on_chunk(lst)