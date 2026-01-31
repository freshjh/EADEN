#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire
cd /opt/data/private/Projects/AIGCDet/DIRE-Variants/ABLATION-STUDY/testsets/DiffusionForensics


python test-part.py

