def BooleanPredictor(tokenizer, model, input_ids, attention_mask, labels):
    """
    get a boolean prediction, whether the token with yes or no gets a higher probability
    """
    # get generated outputs
    generated_outputs = model.generate(input_ids,
                                       return_dict_in_generate=True,
                                       output_scores=True,
                                       output_hidden_states=True)

    # calculate yes token and no token for tokenizer
    yes_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('yes'))[0]
    no_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('no'))[0]

    # for every question check whether the probability for yes token is higher than no
    boolean_predictions = ['yes' if prediction_score[yes_token] >= prediction_score[no_token]
                           else 'no'
                           for prediction_score in generated_outputs.scores[0]
                           ]
    return boolean_predictions
