from pathlib import Path

from tqdm import tqdm
from multiprocessing import Pool
import gzip
import json
import pandas as pd
import os, re
from ExampleGeneration.common.file_utils import CACHE_DIRECTORY, cached_path
from ExampleGeneration.common.multiqa_format_wrapper import MultiQaModel

from ExampleGeneration.common.file_utils import upload_local_file_to_s3


def split(lst, n_groups):
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

def flatten_iterable(listoflists):
    return [item for sublist in listoflists for item in sublist]

def group(lst, max_group_size):
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst) + max_group_size - 1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

def multi_process_lst(lst, apply_on_chunk, chunk_size=1000, n_processes=1, args=None):
    '''
    applies apply_on_chunk on lst using n_processes each gets chunk_size items from lst each time
    '''

    preprocessed_instances = []
    if n_processes == 1:
        yield apply_on_chunk(lst)
    else:
        with Pool(n_processes) as pool:
            chunks = split(lst, n_processes)
            chunks = flatten_iterable(group(c, chunk_size) for c in chunks if len(c) > 0)
            for preproc_inst in pool.imap_unordered(apply_on_chunk,[c for c in chunks]):
                yield preproc_inst

def multi_process_data_stream(s3_input_file, s3_output_file, apply_on_lines_chunk, n_processes, max_chunk_size,
                              copy_header_to_output=True, max_lines_to_process=None, indent_jsonl=False, args=None):

    '''
    applies apply_on_lines_chunk func on s3_input_file lines in multiprocess batches and upload result to s3

    max_chunk_size: max number of lines to send to apply_on_lines_chunk
    copy_header_to_output: whether to copy the first line to the output file
    max_lines_to_process: max number of lines to process
    '''

    # We assume that if output is not meant to be save in s3, then this is a test sample
    # meant for debugging.
    if 's3' in s3_output_file:
        SAVE_TO_S3 = True
    else:
        SAVE_TO_S3 = False


    cache_input_path = cached_path(s3_input_file)
    if ".gz" in s3_input_file:
        input_fp = gzip.open(cache_input_path, 'rb')
    else:
        input_fp = open(cache_input_path, 'r', encoding='utf-8')

    # We assume that if output is not meant to be save in s3, then this is a test sample
    # meant for debugging.
    if SAVE_TO_S3:
        s3_output_file = s3_output_file.replace('s3://', '')
        cache_output_path = os.path.join(CACHE_DIRECTORY, s3_output_file.replace('/', '_'))
        # we assume all s3 file are ".gz"
        output_fp = gzip.open(cache_output_path, 'wb')
    else:
        Path(os.path.dirname(s3_output_file)).mkdir(parents=True, exist_ok=True)
        output_fp = open(s3_output_file, 'w', encoding="utf-8")


    # TODO eigther we are computing the header or not , but let's not just copy it
    if copy_header_to_output and SAVE_TO_S3:
        if 'gz' in s3_output_file:
            output_fp.write(input_fp.readline() )
        else:
            output_fp.write(input_fp.readline().decode())

    # This is used to filter contexts we have not annotated , in final dataset building mode
    if args is not None and args.annotated_questions_file is not None:
        _mturk_rephrased_questions = []
        mturk_rephrased_questions_cached_path = cached_path(args.annotated_questions_file)
        with gzip.open(mturk_rephrased_questions_cached_path) as f:
            header = f.readline()
            for line in f:
                _mturk_rephrased_questions.append(json.loads(line))

    else:
        _mturk_rephrased_questions = None

    all_final_qids = []
    context_ids = []
    pbar = tqdm(ncols=80, smoothing=0.0, unit=" contexts")
    EOF = False
    while not EOF:

        cur_batch = []
        for i in range(n_processes * max_chunk_size):
            context = input_fp.readline()

            if context == b'':
                EOF = True
                break
            else:
                # This is used to filter contexts we have not annotated , in final dataset building mode
                if _mturk_rephrased_questions is not None:
                    if json.loads(context)['id'] not in [c['context_id'] for c in _mturk_rephrased_questions]:
                        continue

                cur_batch.append(context)

        # checks if we processed more than max_number_of_lines_to_process
        if max_lines_to_process and pbar.n + len(cur_batch) >= max_lines_to_process:
            cur_batch = cur_batch[:max_lines_to_process - pbar.n]
            EOF = True

        num_of_contexts = 0
        qas = []
        if args is not None and args.annotated_questions_file is not None:
            context_ids += [json.loads(c)['id'] for c in cur_batch]
        # applies multiprocessing on the current batch
        for processed_batch in multi_process_lst(cur_batch, apply_on_lines_chunk, max_chunk_size, n_processes, args=args):
            for res in processed_batch:

                # appending statistics:
                num_of_contexts += 1

                if type(res) == MultiQaModel:
                    qas += res.qas
                    all_final_qids += [q.qid for q in res.qas]
                #context_ids.append(res.id)

                if SAVE_TO_S3:
                    output_fp.write((str(res) + '\n').encode('utf-8'))
                else:
                    # TODO this is just copied from parse wiki dump, we can do a nicer output than this ...
                    #table_header = res.context[0].table
                    #output_fp.write(res.context[0].url + '\n')
                    #for t_line in table_header:
                    #    line_str = ' | '.join([cell.text for cell in t_line])
                    #    output_fp.write(line_str + '\n')
                    #output_fp.write('\n')
                    if indent_jsonl:
                        s = json.dumps(res.to_json(), sort_keys=True, indent=4, ensure_ascii=False)
                        # just making the answer starts in the sample no have a newline for every offset..
                        s = re.sub('\n\s*(\d+)', r'\1', s)
                        s = re.sub('\n\s*"title"', r'"title"', s)
                        s = re.sub('(\d+)\n\s*]', r'\1]', s)
                        s = re.sub('(\d+)],\n\s*', r'\1],', s)
                        output_fp.write(s + '\n')
                        #output_fp.write((json.dumps(res.to_json(), sort_keys=True, indent=4) + '\n'))
                    else:
                        output_fp.write((json.dumps(res.to_json(), ensure_ascii=False) + '\n'))
                pbar.update(1)

    input_fp.close()
    output_fp.close()
    pbar.close()

    if args is not None and args.annotated_questions_file is not None:
        annotated_questions_df = pd.DataFrame(_mturk_rephrased_questions)
        annotated_questions_df = annotated_questions_df[annotated_questions_df['context_id'].isin(context_ids)]
        annotated_qids = list(annotated_questions_df['qid'])
        ids_not_found = set(annotated_qids) - set(all_final_qids)
        ids_found = set(annotated_qids) & set(all_final_qids)
        #ids_not_types = pd.Series(list(ids_not_found)).str.split('_').str[0].str.split('-').str[0].value_counts()
        ids_not_found_stats = pd.Series(list(ids_not_found)).str.split('_|-').apply(
            lambda x: '_'.join([s for s in x if len(s) > 0 and not any(i.isdigit() for i in s)])).value_counts()
        all_ids_stats = pd.Series(list(annotated_qids)).str.split('_|-').apply(
            lambda x: '_'.join([s for s in x if len(s) > 0 and not any(i.isdigit() for i in s)])).value_counts()
        all_stats = pd.DataFrame(index=all_ids_stats.index)
        all_stats['%'] = ((ids_not_found_stats / all_ids_stats) * 100).fillna(0).astype(int)
        all_stats['#'] = ids_not_found_stats.fillna(0).astype(int)
        all_stats['#'] = all_stats['#'].fillna(0).astype(int)
        all_stats['Total'] = all_ids_stats.fillna(0).astype(int)

        recs_not_found = annotated_questions_df[annotated_questions_df['qid'].isin(ids_not_found)]
        print(f"Details about IDs not found: {recs_not_found[['type', 'date', 'qid', 'context_id']].sort_values('type')}")

        print(f"""{len(ids_found)} IDs matched from annotated question.
        {len(ids_not_found)} IDs not found from contexts present in this chunk:
        Types not found are:\n {all_stats.sort_values(by='#',ascending=False)}""")

    # displaying stats:
    print(f"\nTotal of {num_of_contexts} contexts saved")
    print(f"Total of {len(qas)} questions saved")
    print(f"Question Types\n{pd.Series([q.metadata['type'] for q in qas]).value_counts()}")

    if SAVE_TO_S3:
        upload_local_file_to_s3(cache_output_path, s3_output_file)
        os.remove(cache_output_path)