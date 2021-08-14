def GenerativePredictor(tokenizer, model, input_ids, attention_mask, labels):
    """
    get a generative prediction
    """
    generated_prediction = tokenizer.batch_decode(model.generate(input_ids),
                                                  skip_special_tokens=True)
    return generated_prediction

