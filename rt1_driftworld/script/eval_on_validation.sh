###################### GENERATE

python -m eval.eval_on_validation_shard \
    --config-path configs/sample/rt1_release.yaml --step 29400 \
    --rank 0 --world-size 1 --cfg-scale 2.5 --chunk-gen-frames 8

###################### AGGREGATE

python -m eval.eval_on_validation_aggregate \
    --config-path configs/sample/rt1_release.yaml --step 29400 \
    --world-size 1 --cfg-scale 2.5 --chunk-gen-frames 8
