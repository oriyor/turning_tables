import numpy as np


class SpanEvaluator:
    """
    a class to evaluate mlm task
    """

    def evaluate(self,
                 predictions,
                 output_predictions_path,
                 dataset_name):

        # create a vector for all the predictions
        preds_all = np.array([])
        for p in predictions:
            preds_all = np.append(preds_all, p.correct_predictions.flatten(), axis=0)

        # calculate the different matrices
        span_precision = np.average([p.precision for p in predictions])
        span_f1 = np.average([p.f1 for p in predictions])
        token_em = np.average(preds_all)

        result_dict = {f'{dataset_name}_span_precision': span_precision,
                       f'{dataset_name}_span_f1': span_f1,
                       f'{dataset_name}_token_em': token_em
                       }
        return result_dict
