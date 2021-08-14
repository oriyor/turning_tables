import logging, json
import random

from ExampleGeneration.common.multiqa_format_wrapper import MultiQaModel
from ExampleGeneration.common.multi_process_streaming import multi_process_data_stream
from ExampleGeneration.datajob import DataJob

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


class DictToObject(object):

    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def to_json(self):
        return dict(self.__dict__)

class FormatQuestionsDataJob(DataJob):
    def __init__(self, datajob_name, args):
        self.datajob_name = datajob_name
        logger.info("loading...")
        super().__init__(args)
        self._args = args
        self.output_obj = 's3' in self.output_path

    def format_question(self, c, q):
        # a question is a triplet of question, answer and context
        # we will also add some metadata: context id, page url, question id
        page_title = c.context[0].title.strip()
        table_title = c.context[0].table.table_name.strip()
        page_url = c.context[0].url
        context_prefix = f'In {table_title} of {page_title}: '
        context_content = q.facts + q.distractors
        random.shuffle(context_content)
        facts = '. '.join(context_content)
        context = context_prefix + facts
        return DictToObject({
            'qid': q.qid,
            'question': q.question,
            'phrase': q.question,
            'context': context,
            'answer': ', '.join([str(a) for a in q.answers]),
            'question_metadata': q.metadata,
            'type': q.metadata['type'],
            'template': q.metadata['template'],
            'context_id': c.id,
            'url': page_url,
            'page_title': page_title,
            'table_title': table_title
        })

    def process_chunk(self, contexts):

        random.seed(42)
        questions = []

        contexts = [MultiQaModel.from_json(json.loads(c)) for c in contexts
                    if c]

        # removing extra data from contexts :
        for context in contexts:

            # adding annotation fields to context:
            for q in context.qas:
                formatted_question = self.format_question(context, q)
                if self.output_obj:
                    questions.append(json.dumps(formatted_question.to_json()))
                else:
                    questions.append(formatted_question)
        return questions

    def run_datajob(self, args):
        multi_process_data_stream(self.input_path, self.output_path,
                                  apply_on_lines_chunk=self.process_chunk, n_processes=self._config["n_processes"],
                                  max_chunk_size=self._config["max_chunk_size"],
                                  max_lines_to_process=self._config.get("max_number_of_examples", None),
                                  copy_header_to_output=False,
                                  args=args)
