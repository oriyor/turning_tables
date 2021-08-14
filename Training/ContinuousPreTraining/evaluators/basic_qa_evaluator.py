import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


class BasicQAEvaluator:
    """
    A class with an evaluation loop for QA evaluation
    """
    def evaluate_single_example_method(self):
        """
        method to evaluate a single example
        """
        raise NotImplementedError

    def evaluate(self,
                 predictions,
                 output_predictions_path,
                 dataset_name):
        """
        evaluate a QA dataset
        """
        # todo all this should be in the evaluator, it should get the predictions and return a dict
        # calculate em and f1 for every prediction
        for pred in predictions:
            em_score, f1_score = self.evaluate_single_example_method(pred)
            pred['em'] = em_score
            pred['f1'] = f1_score

        logger.info(f'saving predictions to {output_predictions_path}')

        # save csv with eval
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_predictions_path)

        em = np.average([p['em'] for p in predictions])
        f1 = np.average([p['f1'] for p in predictions])

        result_dict = {f'{dataset_name}_em': em,
                       f'{dataset_name}_f1': f1}

        return result_dict

