CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir /fsx/home/guoyingli/speech/datasets/dump_ljspeech/train/ \
  --outdir /fsx/home/guoyingli/speech/datasets/dump_ljspeech/train/durations/ \
  --checkpoint ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-65000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32
  --win-front 3 \
  --win-back 3
