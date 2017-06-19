
mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir ${HOME}/CAPSTONE_data/Capstone_seq2seq_data/CAPSTONE_results/run7/ \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - ${HOME}/CAPSTONE_data/Capstone_seq2seq_data/test/sources.txt" \
  >  ${HOME}/CAPSTONE_data/Capstone_seq2seq_data/pred/test_predictions.txt
