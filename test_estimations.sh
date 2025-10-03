#!/bin/bash
# Test all valid model/dataset combinations with small particle count
# This is for local testing to ensure everything works

NPART=200
NBLOCKS=2
PE=10000
SEED=1

echo "Testing all model/dataset combinations with npart=$NPART (for quick testing)"
echo "=============================================================================="

# FHP models with all datasets
for k in 0 1 2 3 4 8 10 15 30 40; do
    for dataset in nan 1q 4q; do
        echo ""
        echo "Estimating FHP[k=$k] with dataset=$dataset..."
        uv run python3 estimate_model.py \
            --model "FHP[k=$k]" \
            --dataset "$dataset" \
            --no-slurm \
            --npart $NPART \
            --nblocks $NBLOCKS \
            --pe $PE \
            --seed $SEED

        if [ $? -eq 0 ]; then
            echo "✓ FHP[k=$k] with $dataset completed successfully"
        else
            echo "✗ FHP[k=$k] with $dataset FAILED"
        fi
    done
done

# SI model (not compatible with 1q)
for dataset in nan 4q; do
    echo ""
    echo "Estimating SI with dataset=$dataset..."
    uv run python3 estimate_model.py \
        --model SI \
        --dataset "$dataset" \
        --no-slurm \
        --npart $NPART \
        --nblocks $NBLOCKS \
        --pe $PE \
        --seed $SEED

    if [ $? -eq 0 ]; then
        echo "✓ SI with $dataset completed successfully"
    else
        echo "✗ SI with $dataset FAILED"
    fi
done

echo ""
echo "=============================================================================="
echo "All estimations complete!"
