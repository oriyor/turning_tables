def ListGenerativePredictor(tokenizer, model, input_ids, attention_mask, labels):
    """
    get a list generative prediction with # as the separator
    """
    generated_prediction = tokenizer.batch_decode(model.generate(input_ids),
                                                  skip_special_tokens=True)
    return [pred.split('#')
            for pred in generated_prediction]

