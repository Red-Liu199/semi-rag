CUDA_VISIBLE_DEVICES=0,1,2,3 \
python baseline_faiss.py \
    --params_file "configs_faiss/JSA-20-quac.json" \
    --dataroot "data_quac/rag_format" \
    --model_path "runs_quac/vanilla/JSA-20-test1" \
    --index_path "runs_quac" \
    --skip_cannot_answer \
    --n_gpus 4