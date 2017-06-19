

export VOCAB_SOURCE=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/train/vocab.sources.txt
export VOCAB_TARGET=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/train/vocab.targets.txt
export TRAIN_SOURCES=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/train/sources.txt
export TRAIN_TARGETS=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/train/targets.txt
export DEV_SOURCES=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/dev/sources.txt
export DEV_TARGETS=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/dev/targets.txt
export DEV_TARGETS_REF=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/dev/targets.txt
export TRAIN_STEPS=1000

export MODEL_DIR=${HOME}/CAPSTONE_data/Capstone_seq2seq_data/CAPSTONE_results/run7/
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_bvl.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32\
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
