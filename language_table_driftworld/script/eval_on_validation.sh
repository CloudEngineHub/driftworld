###################### GENERATE
python -m eval.eval_on_validation_shard \
    --config-path configs/sample/language_table_phase2.yaml --step 42800 \
    --rank 0 --world-size 4

python -m eval.eval_on_validation_shard \
    --config-path configs/sample/language_table_phase2.yaml --step 42800 \
    --rank 1 --world-size 4

python -m eval.eval_on_validation_shard \
    --config-path configs/sample/language_table_phase2.yaml --step 42800 \
    --rank 2 --world-size 4

python -m eval.eval_on_validation_shard \
    --config-path configs/sample/language_table_phase2.yaml --step 42800 \
    --rank 3 --world-size 4

###################### AGGREGATE
python -m eval.eval_on_validation_aggregate \
    --config-path configs/sample/language_table_phase2.yaml --step 42800 \
    --world-size 4
