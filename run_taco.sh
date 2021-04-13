export CUDA_VISIBLE_DEVICES=0 
python examples/tacotron2/train_tacotron2.py \
  --train-dir ./dump_ljspeech/train/ \
  --dev-dir ./dump_ljspeech/valid/ \
  --outdir ./examples/tacotron2/exp/train.tacotron2.v1/ \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""
