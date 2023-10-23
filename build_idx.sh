CUDA_VISIBLE_DEVICES=8,9 \
python baseline_faiss.py \
    --params_file "configs_faiss/RAG-5-dstc.json" \
    --knowledge_file "data_dstc/rag_format/knowledge.jsonl" \
    --index_path "runs_dstc" --build_index \
    --n_gpus 2