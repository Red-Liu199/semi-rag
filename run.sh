# CUDA_VISIBLE_DEVICES=8,9 \
# python baseline_faiss.py \
#     --params_file "configs_faiss/RAG-5-quac.json" \
#     --knowledge_file "data_quac/rag_format/knowledge.jsonl" \
#     --index_path "runs_quac" --build_index \
#     --n_gpus 2
CUDA_VISIBLE_DEVICES=8 \
python baseline_faiss.py \
    --params_file "configs_faiss/VRAG-5-quac.json" \
    --dataroot "data_quac/rag_format" \
    --model_path "runs_quac/vanilla/VRAG-5-std-mc" \
    --index_path "runs_quac" \
    --skip_cannot_answer \
    --n_gpus 1 \
    --standard_mc