from ContinuousPreTraining.Common.evaluation_utils import get_drop_metrics
from ContinuousPreTraining.evaluators.basic_qa_evaluator import BasicQAEvaluator


class DropListEval(BasicQAEvaluator):

    def evaluate_single_example_method(self, pred):
        """
        get prediction with max from lists
        """
        max_em_score = 0.0
        max_f1_score = 0.0

        for gold in pred['gold']:
            em_score, f1_score = get_drop_metrics(pred['prediction'], gold)
            max_em_score = max(max_em_score, em_score)
            max_f1_score = max(max_f1_score, f1_score)

        return max_em_score, max_f1_score
