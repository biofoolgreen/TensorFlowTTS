export CUDA_VISIBLE_DEVICES=0 
python3 examples/fastspeech2/train_fastspeech2.py \
  --train-dir ./dump_ljspeech/train/ \
  --dev-dir ./dump_ljspeech/valid/ \
  --outdir ./examples/fastspeech2/exp/train.fastspeech2.v1/bs16/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump_ljspeech/stats_f0.npy \
  --energy-stat ./dump_ljspeech/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""
