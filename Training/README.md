# Turning Tables

## Setup

### Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone https://github.com/oriyor/turning_tables.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd Training
    export PYTHONPATH=${PYTHONPATH}:`pwd`
    ```

3.  Create a virtual environment with Python 3.6 or above:

    ```
    virtualenv venv --python=python3.7 (or python3.7 -m venv venv or conda create -n turningtables python=3.7)
    ```
    
4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use ContinuousPreTraining.

    ```
    source venv/bin/activate (or source venv/bin/activate.csh or conda activate turningtables)
    ```
5.  Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```
    
6.  Download the relevant torch version for your machine (see [torch versions](https://pytorch.org/))


### Downloading PReasM

To download a pre-trained PReasM model please follow the instructions below:

1. Clone the repository

2. Choose a PReasM model:

    ```
    export Sampler=Err/Moment/Uni
    export Size=Base/Large
    ```

3. Download PReasM: 
    ```
    ./bash_scripts/download_preasm.sh
    ```

4. Using PReasM is similar to using T5ForConditionalGeneration in the transformers library:
    ```
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(f'CheckpointsRestored/PReasM-{sampler}-{size}/', return_dict=True)

    ```

### Fine-tune PReasM

To fine-tune PReasM please follow the istructions below:

1. Clone the repository

2. Setup the datasets:
    ```
    ./bash_scripts/setup_datasets.sh
    ```
3. Choose a PReasM model (setting PReasM to False will fine-tune the T5 baseline):

	```
	export PReasM=True/False
	export Sampler=Err/Moment/Uni
	export Size=Base/Large
	export Dataset=drop/iirc/mmqa
	```

4. Fine-tune PReasM. Hyperparameters such as the learning rate and batch size can be updated using the experiment's config file:

     ```
	python ContinuousPreTraining/scripts/train.py -c ContinuousPreTraining/configurations/${Dataset}_config.json --model.PReasM ${PReasM} --model.sampler ${Sampler} --model.size ${Size} 
     ```

5. Verify your results with the official evaluation scripts:
     ```
	python ContinuousPreTraining/scripts/verify_${Dataset}_eval.py --prediction_path /{experiment_name}/{prediction_json_file} --gold_path ContinuousPreTraining/Data/iirc/iirc_dev_drop_format.json
     ```
     
### Training PReasM from scratch

To train PReasM from scratch please follow the instructions below. Note that training PReasM will download data for the original T5 pre-training task and the generated reasoning examples (overall ~13GB). Hyperparameters such as the learning rate and batch size can be updated using the experiment's config file:

1. Clone the repository 
    
2. Set the sampling strategy and model size:

	```
	export SAMPLER=uniform/momentum/errors
	export SIZE=Base/Large
	```

3. Run the following command:

     ```
	 python ContinuousPreTraining/scripts/train.py -c ContinuousPreTraining/configurations/PReasM_${SAMPLER}_config.json --model.size Base -t t5-base -tbs 64 -ebs 128 -gas 1 --training_arguments.eval_steps 5000 --training_arguments.save_steps 5000 --optimizer.lr 1e-4 --experiment.experiment_name PReasM_Base_${SAMPLER}
    ```

### Training the MMQA pipeline

To train the MMQA pipeline retrieval models described in the paper, please follow the instructions below:

1. Clone the repository 
preprocess_mmqa_for_paragraph_classification.py

2. Train the question classifier:
     ```
    python ContinuousPreTraining/scripts/preprocess_mmqa_for_question_classification.py
    python ContinuousPreTraining/scripts/train.py -c ContinuousPreTraining/configurations/mmqa_question_classifier_config.json
     ```

3. Train the paragraph classifier:
     ```
    python ContinuousPreTraining/scripts/preprocess_mmqa_for_paragraph_classification.py
    python ContinuousPreTraining/scripts/train.py -c ContinuousPreTraining/configurations/mmqa_para_classifier_config.json
     ```

4. Unifiy between the classifier's predictions to create the retrieval contexts:
     ```
    python ContinuousPreTraining/scripts/unify_mmqa_context_with_retriever.py.py --dev_questions_classifier_predictions_path {question_classificaiton_dev_predictions.csv} --test_questions_classifier_predictions_path {question_classificaiton_test_predictions.csv} --dev_paragraphs_classifier_predictions_path {paragraph_classificaiton_dev_predictions.csv} --test_paragraphs_classifier_predictions_path {paragraph_classificaiton_test_predictions.csv} 
     ```

5. Train MMQA with the retrieval contexts:
     ```
	python ContinuousPreTraining/configurations/mmqa_retrieval_config.json --model.PReasM ${PReasM} --model.sampler ${Sampler} --model.size ${Size} 
     ```
