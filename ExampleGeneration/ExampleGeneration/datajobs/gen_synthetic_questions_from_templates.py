import json, os
import logging
import gzip
from ExampleGeneration.common.multiqa_format_wrapper import MultiQaModel, Question, Answer, AnswerProperties
from ExampleGeneration.common.multi_process_streaming import multi_process_data_stream
from ExampleGeneration.datajob import DataJob
from ExampleGeneration.question_generators.question_generator_factory import QGenFactory
from ExampleGeneration.common.file_utils import cached_path
from ExampleGeneration.common.question_template_utils import process_question_templates
import pickle

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class GenSyntheticQuestionsFromTemplatesDataJob(DataJob):

    def load_annotated_questions(self, annotated_questions_file):
        # This is used to filter contexts we have not annotated , in final dataset building mode

        if annotated_questions_file is not None:
            self._annotated_questions = []
            mturk_rephrased_questions_cached_path = cached_path(annotated_questions_file)
            with gzip.open(mturk_rephrased_questions_cached_path) as f:
                header = f.readline()
                for line in f:
                    self._annotated_questions.append(json.loads(line))

        else:
            self._annotated_questions = None

    def __init__(self, datajob_name, args, q_templates=None):
        self.datajob_name = datajob_name
        logger.info("loading...")
        super().__init__(args)
        self._args = args

        # get mmqa eval contexts
        self.mmqa_eval_contexts = pickle.load(open("data/mmqa_eval_contexts/eval_contexts.pkl", "rb"))
        # We can also specify annotated_questions_file here via the config
        if 'annotated_questions_file' in self._config and len(self._config['annotated_questions_file']):
            if args.annotated_questions_file is None:
                args.annotated_questions_file = self._config['annotated_questions_file']

        if 'match_annotated_questions' in self._config and self._config['match_annotated_questions']:
            args.match_annotated_questions = True

        # reading questions templates
        if q_templates is None:
            with open(os.path.join('ExampleGeneration', 'question_templates', self._config['question_templates_file'])) as f:
                q_templates = process_question_templates(json.load(f))

            self.q_templates = []
            for template in q_templates:
                if ('templates_to_ignore' not in self._config or \
                    template['name'] not in self._config['templates_to_ignore']) and \
                        ('templates_to_use' not in self._config or \
                         template['name'] in self._config['templates_to_use']):
                    self.q_templates.append(template)

            logger.info(f"Building template: \n{[t['name'] for t in self.q_templates]}")

        else:
            self.q_templates = q_templates

        if 'image_annotation_file' in self._config:
            dummy_ = cached_path(self._config['image_annotation_file'])

        if 'match_annotated_questions' in args and args.match_annotated_questions:
            self.load_annotated_questions(args.annotated_questions_file)
            self._args._annotated_questions = self._annotated_questions

        self._qgens = []
        for template in self.q_templates:
            if template['enable']:
                self._qgens.append(QGenFactory().get_qgen(template['type'], template, self._args))

    def apply_on_chunk(self, contexts):
        contexts = [MultiQaModel.from_json(json.loads(c)) for c in contexts
                    if c]

        # filter mmqa eval set contexts
        contexts = [c for c in contexts
                    if c.id not in self.mmqa_eval_contexts]

        all_questions_text = []
        all_facts_ids = []

        for context in contexts:
            context.new_qas = []

        # construct all question generators:
        for qgen in self._qgens:
            for context in contexts:
                new_qas = []
                questions = qgen.generate(context)

                for question in questions:
                    unique_answers = []

                    answers = question.answers
                    answers_set = set(answers)
                    question.answers = list(answers_set)

                for question in questions:
                    # We make an extra check for duplicate questions here ...
                    if question.question not in all_questions_text:

                        # We can make sure we haven't used these facts in the templates if unique_facts_id is given
                        if question.unique_facts_id != '':
                            if question.unique_facts_id not in all_facts_ids:
                                all_facts_ids.append(question.unique_facts_id)
                                all_questions_text.append(question.question)
                                new_qas.append(question)

                        else:
                            all_questions_text.append(question.question)
                            new_qas.append(question)

                context.new_qas += new_qas

        # filtering contexts (here context with no question
        filtered_contexts = []
        for c in contexts:
            if 'reset_qas' in self._config and self._config['reset_qas']:
                c.qas = c.new_qas
            else:
                c.qas += c.new_qas
            del c.new_qas

            if len(c.qas) > 0:
                if 'question_type_to_keep' in self._config:
                    c.qas = [q for q in c.qas if q.metadata['type'] in self._config['question_type_to_keep']]
                filtered_contexts.append(c)

        return filtered_contexts

    def run_datajob(self, args, indent_jsonl=False):
        multi_process_data_stream(self.input_path, self.output_path,
                                  apply_on_lines_chunk=self.apply_on_chunk, n_processes=self._config['n_processes'],
                                  max_chunk_size=self._config['max_chunk_size'], indent_jsonl=indent_jsonl,
                                  max_lines_to_process=self._config['max_number_of_examples'], args=args)
