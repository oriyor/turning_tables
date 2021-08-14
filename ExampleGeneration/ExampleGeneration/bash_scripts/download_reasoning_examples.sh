echo "Downloading reasoning examples to Data/reasoning_examples ..."
echo "Downloading train reasoning examples to Data/reasoning_examples/train ..."

mkdir -p reasoning_examples
cd reasoning_examples

mkdir -p train
cd train

wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/arithmetic_addition.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/arithmetic_superlatives.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/composition_2_hop.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/composition.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/conjunction.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/counting.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/every_quantifier.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/most_quantifier.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/numeric comparison.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/numeric_comparison_boolean.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/numeric_superlatives.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/only_quantifier.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/temporal_comparison_boolean.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/temporal_comparison.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/temporal_difference.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_train/temporal_superlatives.gz

cd ..

echo "Downloading train reasoning examples to Data/reasoning_examples/dev ..."
mkdir -p dev
cd dev

wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/arithmetic_addition.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/arithmetic_superlatives.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/composition_2_hop.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/composition.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/conjunction.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/counting.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/every_quantifier.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/most_quantifier.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/numeric comparison.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/numeric_comparison_boolean.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/numeric_superlatives.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/only_quantifier.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/temporal_comparison_boolean.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/temporal_comparison.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/temporal_difference.gz
wget https://tabreas.s3.us-west-2.amazonaws.com/generated_reasoning_examples_dev/temporal_superlatives.gz
cd ../..