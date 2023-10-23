CUDA_VISIBLE_DEVICES=8 \
python evaluate.py \
    --params_file "configs_faiss/JSA-20-quac-test.json" \
    --dataroot "data_quac/rag_format" \
    --index_path "runs_quac" \
    --eval_only \
    --eval_dataset test