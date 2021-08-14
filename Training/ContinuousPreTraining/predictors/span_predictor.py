import torch
import numpy as np
from typing import Set


class SpanPrediction:
    """
    span prediction object
    """
    def __init__(self,
                 correct_predictions,
                 tokens_predictions_dict,
                 tokens_labels_dict,
                 precision,
                 f1):
        """
        init relevant fields for a span prediction
        """
        self.correct_predictions = correct_predictions
        self.tokens_predictions_dict = tokens_predictions_dict
        self.tokens_labels_dict = tokens_labels_dict
        self.precision = precision
        self.f1 = f1


def _compute_f1(predicted_bag: Set[int], gold_bag: Set[int]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def get_span_prediction_dict(predictions):
    """
    method to get dictionary between spans and their predictions
    """
    span_prediction_dict = {}
    i = 0
    num_predicted_tokens = len(predictions)
    last_extra_id = None

    # iterate over all tokens
    while i < num_predicted_tokens:
        pred_token = predictions[i]

        # check if this is an extra id token
        if 32000 <= pred_token <= 33000:
            span_prediction_dict[pred_token] = []
            last_extra_id = pred_token

        else:
            if last_extra_id is not None:
                span_prediction_dict[last_extra_id].append(pred_token)

        # raise cnt for i
        i += 1
    return span_prediction_dict


def get_precision(gold_spans, predicted_spans):
    """
    calculate span precision
    """
    # make sure we don't calculate by zero
    if len(gold_spans) == 0:
        return 0

    num_spans = len(gold_spans)
    correct_predictions = 0
    for span_key in gold_spans.keys():
        if span_key in predicted_spans:
            if predicted_spans[span_key] == gold_spans[span_key]:
                correct_predictions += 1

    return correct_predictions / num_spans


def get_average_f1(gold_spans, predicted_spans):
    """
    get average f1 between a gold and predicted span
    """
    # make sure we don't calculate by zero
    if len(gold_spans) == 0:
        return 0

    num_spans = len(gold_spans)
    tot_f1 = 0
    for span_key in gold_spans.keys():
        if span_key in predicted_spans:
            tot_f1 += _compute_f1(set(gold_spans[span_key]), set(predicted_spans[span_key]))

    return tot_f1 / num_spans


def SpanPredictor(tokenizer, model, input_ids, attention_mask, labels):
    """
    span predictor for mlm task
    """

    # init fields
    span_predictions = []
    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs.logits

    # mlm preds
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    correct_preds = (preds == labels_cpu)

    # calculate the precision and f1 for every sample
    for k, pred in enumerate(preds):
        preds_spans_dict = get_span_prediction_dict(preds[k])
        labels_spans_dict = get_span_prediction_dict(labels_cpu[k])
        precision = get_precision(preds_spans_dict, labels_spans_dict)
        f1 = get_average_f1(preds_spans_dict, labels_spans_dict)

        # append prediction
        span_predictions.append(SpanPrediction(correct_predictions=correct_preds,
                                               tokens_predictions_dict=preds_spans_dict,
                                               tokens_labels_dict=labels_spans_dict,
                                               precision=precision,
                                               f1=f1))

    return span_predictions
