echo 'Running Compression Visualization'

python Demo.py \
    --model_path Qwen/Qwen3-8B \
    --in_jsonl ./QA.jsonl \
    --out_json ./QA_Result.json \
    --dtype bf16 \
    --page_size 64 \
    --anchor_pages 1 \
    --flow_window 1 \
    --flash_top_k 1

python Build_Html.py

echo 'All done. Visualization file is QA_Report.html'