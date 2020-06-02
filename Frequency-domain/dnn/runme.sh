#! /bin/bash
# Author: Hao Shi (Tianjin University, China)

step=0

workspace=./workspace/     # workspace is used to store some necessary files lists
mkdir -p $workspace

# Prameters for Feature Extraction
num_concat=7
normalization=1
sample_rate=8000
stft_size=256
stft_shift=128



######   NOTE for STEP 1: Feature Extraction
if [ $step -le 0 ]; then
    # python -u feature_extraction.py --num_concat --normalization --datasets --shuffles --stft_size --stft_shift --sample_rate
    python -u feature_extraction.py ${num_concat} 1 tr 1 ${stft_size} ${stft_shift} ${sample_rate} &
    python -u feature_extraction.py ${num_concat} 1 va 1 ${stft_size} ${stft_shift} ${sample_rate} &
    python -u feature_extraction.py ${num_concat} 1 tt 0 ${stft_size} ${stft_shift} ${sample_rate} &

    wait
fi