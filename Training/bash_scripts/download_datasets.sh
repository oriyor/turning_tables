# download drop
cd ./ContinuousPreTraining/Data
mkdir drop
cd ./drop
echo "downloading DROP ..."
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip
unzip drop_dataset.zip
mv ./drop_dataset/*.json .
mv drop_dataset.zip drop_dataset
rm -rf drop_dataset
cd ../../..

# download IIRC
cd ./ContinuousPreTraining/Data
echo "downloading IIRC ..."
wget http://jamesf-incomplete-qa.s3.amazonaws.com/iirc.tar.gz
tar -xzf iirc.tar.gz
rm iirc.tar.gz

echo "downloading IIRC dev in drop format for eval script (see https://github.com/jferguson144/IIRC-baseline)..."
cd ./iirc
wget https://tabreas.s3.us-west-2.amazonaws.com/iirc/iirc_dev_drop_format.json
cd ../../..

# download MMQA
cd ./ContinuousPreTraining/Data
echo "downloading MMQA ..."
mkdir mmqa
cd ./mmqa
wget https://github.com/allenai/multimodalqa/blob/master/dataset/MMQA_train.jsonl.gz?raw=true
wget https://github.com/allenai/multimodalqa/blob/master/dataset/MMQA_dev.jsonl.gz?raw=true
wget https://github.com/allenai/multimodalqa/blob/master/dataset/MMQA_test.jsonl.gz?raw=true
wget https://github.com/allenai/multimodalqa/blob/master/dataset/MMQA_texts.jsonl.gz?raw=true
wget https://github.com/allenai/multimodalqa/blob/master/dataset/MMQA_tables.jsonl.gz?raw=true

echo "downloading preprocessed contexts after pipeline retrieval..."
wget https://tabreas.s3.us-west-2.amazonaws.com/mmqa/parsed_mmqa_dev_retrieval.json

cd ../../..