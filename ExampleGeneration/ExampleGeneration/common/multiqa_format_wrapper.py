import json
import hashlib
from copy import copy
import random

random.seed(42)


class Fact:
    """
    basic fact class
    """

    def __init__(self, table_title, page_title, page_url,
                 src_column_ind, target_column_ind, source_val_indices,
                 src_column_header, target_column_header,
                 src_column_value, target_column_values,
                 is_numeric, is_datetime, filtered):
        self.table_title = table_title
        self.page_title = page_title
        self.page_url = page_url
        self.src_column_ind = src_column_ind
        self.target_column_ind = target_column_ind
        self.source_val_indices = source_val_indices
        self.src_column_header = src_column_header
        self.formatted_src_column_header = src_column_header.lower().replace('app.', 'appearances') \
            .replace('no.', 'number').replace('pos.', 'position')

        self.target_column_header = target_column_header
        self.formatted_target_column_header = target_column_header.lower().replace('app.', 'appearances').replace('no.',
                                                                                                                  'number').replace(
            'pos.', 'position').replace('pop.', 'population').replace('avg.', 'average')

        # replace frequent columns
        if self.formatted_target_column_header == 'pos':
            self.formatted_target_column_header = 'position'

        if self.formatted_target_column_header == 'pts':
            self.formatted_target_column_header = 'points'

        self.src_column_value = src_column_value
        self.target_column_values = target_column_values
        self.src_column_header = src_column_header
        self.is_numeric = is_numeric
        self.is_datetime = is_datetime
        self.filtered = filtered

        # we will create a unique id for the fact
        m = hashlib.md5()
        m.update(page_url.encode())
        m.update(str(target_column_ind).encode())
        m.update(str(src_column_ind).encode())
        m.update(src_column_value.encode())
        self.fact_id = m.hexdigest()

    def format_fact(self, date_time=False, filter_duplicate_facts=True):

        # remove duplicates and shuffle
        if filter_duplicate_facts:
            target_vals = list(set(self.target_column_values))
        else:
            target_vals = self.target_column_values
        random.shuffle(target_vals)

        # split between list and singleton target values
        if len(target_vals) > 1:

            # we won't use commas for lists of length 2
            if len(target_vals) == 2:
                target_cells_str = target_vals[0] + ' and ' + target_vals[1]
            else:
                target_cells_str = ', '.join(target_vals[:-1]) + ', and ' + target_vals[-1]

            target_column_suffix = 's' if self.target_column_header[-1] != 's' else ''
            return f'The {self.formatted_target_column_header}{target_column_suffix}' \
                   f' when the {self.formatted_src_column_header} was {self.src_column_value}' \
                   f' were {target_cells_str}'

        else:

            target_cells_str = target_vals[0].strip()

            # format for datetime facts
            if date_time:
                return f'The {self.formatted_src_column_header} was {self.src_column_value}' \
                       + ' in ' + target_cells_str

            # else continue

            return f'The {self.formatted_target_column_header}' \
                   f' when the {self.formatted_src_column_header} was {self.src_column_value}' \
                   f' was {target_cells_str}'

    def format_datetime_fact(self, date_range):
        """
        format date range facts
        """
        date_fact_prefix = f'The {self.src_column_header} was {self.src_column_value}'

        # we check whether to return a single value or a span
        if date_range.min_date == date_range.max_date:
            return date_fact_prefix + ' in ' + date_range.max_date

        else:
            return date_fact_prefix + ' between ' + date_range.min_date + ' and ' + date_range.max_date

    def to_json(self):
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)

    def __str__(self):
        return json.dumps(self.to_json())


class SyntheticQuestion:
    def __init__(self, qid, question, answers, facts, distractors, metadata={}, facts_ids=None, unique_facts_id=None):
        self.qid = qid
        self.question = question
        self.facts = facts
        self.answers = answers
        self.distractors = distractors
        self.metadata = metadata

        # check if we can assign a unique id to the question's facts
        self.unique_facts_id = ''
        if facts_ids is not None:
            self.unique_facts_id = self.metadata['type'] + '_'.join(sorted(facts_ids))

    def to_json(self):
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)


class WikiTableColumnDescriptor:
    def __init__(self, column_name, metadata={}):
        self.column_name = column_name
        self.metadata = metadata

    def to_json(self):
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)


class WikiTable:
    def __init__(self, table_rows=[[]], table_name="", header=[], metadata={}):
        self.table_rows = table_rows
        self.table_name = table_name
        self.header = header
        self.metadata = metadata

    def to_json(self):
        as_json = dict(self.__dict__)
        as_json["header"] = [column_descriptor.to_json() for column_descriptor in self.header]
        table = []
        for row in self.table_rows:
            table.append([c.to_json() for c in row])
        as_json["table_rows"] = table
        return as_json

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        'format header'
        j["header"] = [WikiTableColumnDescriptor.from_json(column_descriptor) for column_descriptor in j["header"]]

        'format table'
        table = []
        for row in j["table_rows"]:
            table.append([TableCell.from_json(c) for c in row])
        j["table_rows"] = table

        j["metadata"] = j.get("metadata", {})

        return cls(**j)


class ModelEntity:
    def to_json(self):
        as_json = dict(self.__dict__)
        for k, v in as_json.items():
            if isinstance(v, ModelEntity):
                as_json[k] = v.to_json()
            elif type(v) == list:
                for i, item in enumerate(v):
                    if isinstance(item, ModelEntity):
                        v[i] = item.to_json()
        return as_json

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)

    def __str__(self):
        return json.dumps(self.to_json())


class Doc:
    def __init__(self, title, url, id=None, text=None, table=None, metadata=None, snippet=None, image=None, caption=None):
        self.title = title
        self.url = url
        if id is None and metadata is not None:
            self.id = metadata['id']
        elif id is None:
            # generating an ID using url:
            m = hashlib.md5()
            m.update(str(url).encode())
            self.id = m.hexdigest()
        else:
            self.id = id
        if text is not None:
            self.text = text
        if table is not None:
            self.table = table
        if metadata is not None:
            self.metadata = metadata
        if snippet is not None:
            self.snippet = snippet
        if image is not None:
            self.image = image

    def to_json(self):
        as_json = dict(self.__dict__)
        # for context that are not table deleted the table entry
        if hasattr(self, 'table'):
            as_json["table"] = self.table.to_json()

        # deleting empty fields
        # keys = list(as_json.keys())
        # for k in keys:
        #    if len(as_json[k]) == 0:
        #        del as_json[k]
        return as_json

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        if "table" in j:
            j["table"] = WikiTable.from_json(j["table"])
        # TODO this is for backward compatability...
        if 'caption' in j:
            del j['caption']

        # TODO this is for backward compatability...
        if 'id' not in j:
            if 'id' in j['metadata']:
                j['id'] = j['metadata']['id']
            else:
                # generating an ID using url:
                m = hashlib.md5()
                m.update(str(j['url']).encode())
                j['id'] = m.hexdigest()
        return cls(**j)

    def __str__(self):
        return json.dumps(self.to_json())


class TableCell:

    def __init__(self, text, links):
        self.text = text
        self.links = links

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        j["links"] = [WikiLink.from_json(l) for l in j["links"]]
        return cls(**j)

    def to_json(self):
        as_json = dict(self.__dict__)
        as_json["links"] = [l.to_json() for l in self.links]
        return as_json

    def __str__(self):
        return json.dumps(self.to_json())


class WikiLink:

    def __init__(self, text, wiki_title, url=None):
        self.text = text
        self.wiki_title = wiki_title
        self.url = "https://en.wikipedia.org/wiki/{}".format(wiki_title).replace(' ', '_') \
            if not url else url

    def to_json(self):
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)

    def __str__(self):
        return json.dumps(self.to_json())


class Answer:
    def __init__(self, answer, type, modality, is_extractive, aliases=[], text_instances=[], table_indices=[]):
        # yesno or generated_text or date or number or extractive
        self.answer = answer
        self.type = type
        self.modality = modality
        self.is_extractive = is_extractive
        # optional fields
        if len(aliases) > 0:
            self.aliases = aliases
        self.text_instances = text_instances
        self.table_indices = table_indices

    def to_json(self):
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)

    def __str__(self):
        return json.dumps(self.to_json())


class AnswerProperties:
    def __init__(self, sequence_format, annotators_answer_candidates=[], evaluation_method=None,
                 cannot_answer="not_available"):
        self.cannot_answer = cannot_answer
        self.sequence_format = sequence_format
        if len(annotators_answer_candidates) > 0:
            self.annotators_answer_candidates = annotators_answer_candidates
        if evaluation_method is not None:
            self.evaluation_method = evaluation_method

    def to_json(self):
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        return cls(**j)

    def __str__(self):
        return json.dumps(self.to_json())


class Question:
    def __init__(self, qid, question, answers, answer_properties, metadata={}, supporting_context=[]):
        self.qid = qid
        self.question = question
        self.answer_properties = answer_properties
        self.answers = answers
        self.metadata = metadata
        self.supporting_context = supporting_context

    @classmethod
    def convert_from_old_version(self, old_answers):
        if 'list' in old_answers:
            answer_properties = AnswerProperties(sequence_format='list')
            new_answers = []
            for answer in old_answers['list']:
                if type(answer['extractive']['instances'][0]) == list:
                    answer['extractive']['instances'] = answer['extractive']['instances'][0]
                new_answers.append(Answer(answer=answer['extractive']['answer'], \
                                          text_instances=answer['extractive']['instances'],
                                          modality='text', type='string', is_extractive=True))

        elif 'single_answer' in old_answers:
            single_answer = old_answers['single_answer']
            answer_properties = AnswerProperties(sequence_format='single_answer')
            if 'extractive' in single_answer:
                new_answers = [Answer(answer=single_answer['extractive']['answer'], \
                                      text_instances=single_answer['extractive']['instances'],
                                      modality='text', type='string', is_extractive=True)]
            else:
                new_answers = [Answer(answer=single_answer['yesno']['answer'], \
                                      modality='text', type='yesno', is_extractive=False)]
        else:
            raise (ValueError)

        return new_answers, answer_properties

    @classmethod
    def from_json(cls, j):
        j = copy(j)

        # supporting old version answer format, ASSUMING ONLY USED WITH NQ QUESTIONS!
        if 'list' in j["answers"] or 'single_answer' in j["answers"]:
            j["answers"], j["answer_properties"] = cls.convert_from_old_version(j["answers"])
        else:
            answers = []
            for answer in j["answers"]:
                answers.append(Answer.from_json(answer))
            j["answers"] = answers

            # TODO this is temporary until we fix the format in all files:
            if "answer_properties" in j:
                j["answer_properties"] = AnswerProperties.from_json(j["answer_properties"])
            else:
                j["answer_properties"] = None

        return cls(**j)

    def to_json(self):
        as_json = dict(self.__dict__)
        as_json["answers"] = [a.to_json() for a in self.answers]
        as_json["answer_properties"] = self.answer_properties.to_json()
        return as_json

    def __str__(self):
        return json.dumps(self.to_json())


def answer_type(answer):
    if str(answer).isdigit():
        return "number"
    return "extractive"


def extract_answers(answer):
    if 'generated_text' in answer:
        return answer['generated_text'], False
    if 'number' in answer:
        return answer['number'], False
    if 'extractive' in answer:
        return answer['extractive']['answer'], True, answer['extractive'].get("instances", None), False
    if 'yesno' in answer:
        return answer['yesno']["answer"], False, [], True


class MultiQaModel:

    def __init__(self, id, context, qas, candidate_qas=[]):
        self.qas = qas
        self.context = context
        self.id = id
        self.candidate_qas = candidate_qas

    @classmethod
    def from_json(cls, j):
        j = copy(j)
        context = [Doc.from_json(doc) for doc in j["context"]["documents"]]
        qas = [SyntheticQuestion.from_json(q) for q in j["qas"]]
        # In this project we use candidate_qas, that is identical to qas, but used only for
        # candidate questions, that will be removed in the final release dataset.
        if "candidate_qas" in j:
            candidate_qas = [Question.from_json(q) for q in j["candidate_qas"]]
        else:
            candidate_qas = []
        return MultiQaModel(j["id"], context, qas, candidate_qas)

    def to_json(self):
        as_dict = dict(self.__dict__)
        as_dict["context"] = {"documents": [d.to_json() for d in self.context]}
        as_dict["qas"] = [q.to_json() for q in self.qas]
        if len(self.candidate_qas) > 0:
            as_dict["candidate_qas"] = [q.to_json() for q in self.candidate_qas]
        elif "candidate_qas" in as_dict:
            del as_dict["candidate_qas"]
        return as_dict

    def __str__(self):
        return json.dumps(self.to_json())
