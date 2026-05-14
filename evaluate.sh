#!/bin/bash

python evaluate.py \
    --model_path '/home/hep/tlt26/RW_Snakemake/saved_models/6D_mode1_11_12_13/6D/XGB_model_6D.pkl' \
    --original_test_nD '/home/hep/tlt26/RW_Snakemake/saved_samples/6D_5andmode_mode_13/6D/original_test.csv' \
    --original_test '/home/hep/tlt26/RW_Snakemake/saved_samples/6D_5andmode_mode_13/21D/original_test.csv' \
    --target_test_nD '/home/hep/tlt26/RW_Snakemake/saved_samples/6D_5andmode_mode_13/6D/target_test.csv' \
    --target_test '/home/hep/tlt26/RW_Snakemake/saved_samples/6D_5andmode_mode_13/21D/target_test.csv' \
    --swd_distribution_target_3D '/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_13/3D/swd_distribution_3D.npy' \
    --swd_distribution_target_8D '/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_13/8D/swd_distribution_8D.npy' \
    --swd_distribution_target_21D '/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_13/21D/swd_distribution_21D.npy' \
    --swd_distribution_target_nD '/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_13/6D/swd_distribution_6D.npy' \
    --binning_file '/home/hep/tlt26/RW_Snakemake/binnings.json' \
    --output_path '/home/hep/tlt26/RW_Snakemake/Saved_evaluation/trained_1_11_12_13_eval_13/'