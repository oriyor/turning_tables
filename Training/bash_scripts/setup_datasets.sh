echo "Downloading datasets..."
./bash_scripts/download_datasets.sh

echo "Preprocessing drop..."
python ./ContinuousPreTraining/scripts/preprocess_drop.py

echo "Preprocessing mmqa..."
python ./ContinuousPreTraining/scripts/preprocess_mmqa.py