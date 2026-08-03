###################### GENERATE

# Rank 0
python -m eval.eval_on_validation_shard \
    --config-path configs/sample/bridge_release.yaml --step 55200 \
    --rank 0 --world-size 8 --cfg-scale 3.0

# Similarly run for the other ranks

###################### AGGREGATE

python -m eval.eval_on_validation_aggregate \
    --config-path configs/sample/bridge_release.yaml --step 55200 \
    --world-size 8 --cfg-scale 3.0
