echo "Downloading PReasM to CheckpointsRestored/PReasM-$Sampler-$Size ..."

mkdir -p CheckpointsRestored
cd CheckpointsRestored

mkdir -p PReasM-$Sampler-$Size
cd PReasM-$Sampler-$Size

echo "Downloading config.json..."
wget https://tabreas.s3.us-west-2.amazonaws.com/PReasM/PReasM-$Sampler-$Size/config.json

echo "Downloading optimizer.pt..."
wget https://tabreas.s3.us-west-2.amazonaws.com/PReasM/PReasM-$Sampler-$Size/optimizer.pt

echo "Downloading pytorch_model.bin..."
wget https://tabreas.s3.us-west-2.amazonaws.com/PReasM/PReasM-$Sampler-$Size/pytorch_model.bin

echo "Downloading scheduler.pt..."
wget https://tabreas.s3.us-west-2.amazonaws.com/PReasM/PReasM-$Sampler-$Size/scheduler.pt

echo "Downloading trainer_state.json..."
wget https://tabreas.s3.us-west-2.amazonaws.com/PReasM/PReasM-$Sampler-$Size/trainer_state.json

echo "Downloading training_args.bin..."
wget https://tabreas.s3.us-west-2.amazonaws.com/PReasM/PReasM-$Sampler-$Size/training_args.bin

cd ../..
