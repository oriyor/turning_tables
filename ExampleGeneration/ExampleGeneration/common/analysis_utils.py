from ExampleGeneration.common.file_utils import cached_path
from ExampleGeneration.common.multiqa_format_wrapper import MultiQaModel
import gzip, json, random
import pandas as pd
import logging
import urllib

from ExampleGeneration.common.table_wrapper import WikiTable

def dump_synthetic_questions_analysis(input_file, output_file, dump_csv=True):
    """
    This function dumps a sample of questions for manual analysis
    If output_file is csv dumps a csv, otherwise a jsonl

    output csv columns: qid, question, answers, context_url, context_title, table_title, link_answer

    :param input_file: contexts file path (can be either s3 or local, jsonl or zipped jsonl)
    :param output_file: local path to dump the sample result
    :param sample_size: number of questions to sample
    """
    cache_input_path = cached_path(input_file)
    if ".gz" in input_file:
        input_fp = gzip.open(cache_input_path, 'rb')
    else:
        input_fp = open(cache_input_path, 'r', encoding='utf8')

    questions_pool = []
    for q in input_fp:
        questions_pool.append(json.loads(q))

    df = pd.DataFrame(questions_pool)

    # dump csv if local
    if dump_csv:
        df.to_csv(output_file, encoding='utf-8-sig', index=False)

    # print stats
    num_questions  = df.shape[0]
    num_contexts = len(df.groupby('context_id'))
    print(f'Generated {num_questions} questions from {num_contexts} contexts')
    print()
    print('### Type Breakdown ###')
    print(df.groupby('type').size())
    print()
    print('### Template Breakdown ###')
    print(df.groupby('template').size())

def dump_manual_analysis_facts(input_file, output_file, agg_column=None, c_sample_size=3, q_sample_size=None):
    """
    This function dumps a sample of questions for manual analysis
    If output_file is csv dumps a csv, otherwise a jsonl

    output csv columns: qid, question, answers, context_url, context_title, table_title, link_answer

    :param input_file: contexts file path (can be either s3 or local, jsonl or zipped jsonl)
    :param output_file: local path to dump the sample result
    :param sample_size: number of questions to sample
    """
    cache_input_path = cached_path(input_file)
    if ".gz" in input_file:
        input_fp = gzip.open(cache_input_path, 'rb')
    else:
        input_fp = open(cache_input_path, 'r', encoding='utf8')

    questions_pool = []
    for multimodal_j in input_fp:
        if multimodal_j == b'\n':
            continue

        multimodal_obj = MultiQaModel.from_json(json.loads(multimodal_j))

        table = multimodal_obj.context[0]

        context_data = {
            "context_id": multimodal_obj.id,
            "context_url": table.url,
            "context_title": table.title,
            "table_title": table.table.table_name,
            "key_column": WikiTable(multimodal_obj).get_key_column()
        }

        for qas in multimodal_obj.qas:
            qas_j = qas.to_json()
            qas_j["num_distractors"] = len(qas_j['distractors'])
            qas_j["type"] = qas_j['metadata']['type']
            qas_j["reasoning"] = ' '.join(qas_j['metadata']['reasoning'])
            qas_j["answer_type"] = qas_j['metadata']['answer_type']
            qas_j["reversed_facts"] = qas_j['metadata']['reversed_facts']
            qas_j["template"] = qas_j['metadata']['template']

            qas_j.update(context_data)
            questions_pool.append(qas_j)


    if len(questions_pool) == 0:
        return

    if q_sample_size and q_sample_size < len(questions_pool):
        sampled_questions = random.sample(questions_pool, k=q_sample_size)
    else:
        sampled_questions = questions_pool

    logging.info(f"dumping sample to {output_file}")
    if output_file.endswith("csv"):
        df = pd.DataFrame(sampled_questions)

        # unwinding the metadata:
        all_keys = set([item for id,r in df.iterrows() for item in list(r['metadata'].keys())])
        for k in all_keys:
            df[k] = [r['metadata'][k] if k in r['metadata'] else None for id,r in df.iterrows()]

        selected_cols = ['qid', 'reasoning', 'type', 'template', 'answer_type', 'question', 'answers', 'facts', 'reversed_facts', 'distractors', 'num_distractors']
        selected_cols += ['context_url', 'context_title', 'table_title','context_id','key_column']

        # Here we just aggregate by a certain column such as table_title
        # and sample the first three results...
        if agg_column:
            selected_cols = ['context_count', 'count', agg_column] + selected_cols
            aggregation_counts = df[agg_column].value_counts()
            agg_df = pd.DataFrame()
            for val, count in aggregation_counts.iteritems():

                val_df = df[df[agg_column] == val]
                val_df['context_count'] = len(set(val_df['context_url']))
                if len(val_df)>c_sample_size:
                    val_df = val_df.sample(n=c_sample_size)
                val_df['count'] = count
                agg_df = agg_df.append(val_df,ignore_index=True)
            df = agg_df

        #df = df.drop(["metadata"], axis=1)
        df = df[selected_cols]

        df.to_csv(output_file, encoding='utf-8-sig', index=False)

    else:  # Dump as jsonl
        with open(output_file, 'w', encoding='utf-8' ) as f:
            for q in sampled_questions:
                f.write(json.dumps(q, indent=3, ensure_ascii=False) + '\n')
