import json
import gzip
from datasets import tqdm


class OrderedBlock:
    """
    Ordered block object, which contains a block of sequences of similar size, and the maximum sequence length
    """
    def __init__(self, block, max_seq):
        self.block = block
        self.max_seq = max_seq


def build_data_blocks(data_path, max_block_size):
    """
    Method to build blocks of data with similar size
    data_path: path to data path, each example must contain a phrase and a context
    max_block_size: max number of examples in a single block
    """
    input_examples = []
    with gzip.open(data_path, "r") as f:
        for i, l in enumerate(tqdm(f)):
            input_examples.append(json.loads(l))

    if 'phrase' in input_examples[0]:
        input_examples.sort(key=lambda x: len(x['phrase']) + len(x['context']))
    else:
        input_examples.sort(key=lambda x: len(x['context']))
    ordered_blocks = []

    while len(input_examples) > 0:
        to_take = min(max_block_size, len(input_examples))
        # select = random.randint(0, len(input_examples) - to_take)
        select = 0
        block = input_examples[select:select + to_take]
        if 'phrase' in block[0]:
            max_seq = max([len(x['phrase']) + len(x['context']) for x in
                       block])
        else:
            max_seq = max([len(x['context']) for x in block])
        ordered_blocks.append(OrderedBlock(block=block,
                                           max_seq=max_seq))
        del input_examples[select:select + to_take]
    return ordered_blocks
