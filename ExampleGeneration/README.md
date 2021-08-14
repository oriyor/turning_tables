

## Setup

## Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone https://github.com/oriyor/turning_tables.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd ExampleGeneration
    export PYTHONPATH=${PYTHONPATH}:`pwd`
    ```

3.  Create a virtual environment with Python 3.6 or above:

    ```
    virtualenv venv --python=python3.7 (or python3.7 -m venv venv or conda create -n turning python=3.7)
    ```

4.  Activate the virtual environment:
    ```
    source venv/bin/activate (or source venv/bin/activate.csh or conda activate turning)
    ```
5.  Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Example Generation

Our pre-precossed data is split to 35 chunks containing 20K tables each. Please note that running on all chunks can take time and speed-up is possible with multi-processing (~10 hours with 20 processes). The examples are generated using a pipeline of datajobs which are logical units that perform actions on pre-processed Wikipedia tables. The pipeline includes 3 datajobs:
* ReasClassifyColumnTypes: classifies the table's columns to semantic types
* GenQuestionsFromTemplates_TabReas: generates examples from the pre-processed tables 
* FormatSyntheticQuestions: post-processes the examples to pseudo-language question, context, answer triplets 

Please see `ExampleGeneration/configurations/config_reas.json` for the full configuration of each datajob including the number of processes and the path to the input data. To generate examples:
   
   1.  Choose start and end chunks between 0 and 34:
```
export Start_Chunk=0
export End_Chunk=0 
```

   3.  Generate examples:

    python ExampleGeneration/run_multiple_chunks.py -config config_reas.json -dj ReasClassifyColumnTypes,GenQuestionsFromTemplates_TabReas,FormatSyntheticQuestions     -wd data/data_chunks/ -sc ${Start_Chunk}  -ec ${End_Chunk}

## Downloading generated reasoning examples

Our generated examples are publicly available. To download the examples:

1. Clone the repository

2. Download the examples:
    ```
	./ExampleGeneration/bash_scripts/download_reasoning_examples.sh 
    ```
    
## Other

A caching infra is used, so please make sure to have enough disk space and control the cache directory using `TURNINGTABLES_CACHE_ROOT` env variable.

## Parsing tables from a Wikipedia dump

Our infra supports parsing a Wikipedia dump to tables based on [WikiExtractor](https://github.com/attardi/wikiextractor) and  [WikiTextParser](https://github.com/5j9/wikitextparser). To parse a full Wikipedia dump, see `ExampleGeneration/ExampleGeneration/bash_scripts/parse_wiki_dump.sh`. 
