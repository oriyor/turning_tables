from ContinuousPreTraining.Common.evaluation_utils import get_drop_metrics
from ContinuousPreTraining.evaluators.basic_qa_evaluator import BasicQAEvaluator


class DropEval(BasicQAEvaluator):

    def evaluate_single_example_method(self, pred):
        em_score, f1_score = get_drop_metrics(pred['prediction'], pred['gold'])
        return em_score, f1_score
