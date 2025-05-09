#!/bin/bash

# Define common variables
EXPERIMENT_NAME="detrend_reg_extended_winter_psl_zg_grad"
X_FILE_POSTFIX="detrended"
Y_FILE="CMIP6_target_combined_detrended.nc"   
VARS="psl,zg_grad"
months="10,11,0,1,2"  # NDJFM

# Parse command line argument for retraining
RETRAIN=${2:-false}  # Default to false if no argument provided
GPU_IDX=${1:-0}  # Default to GPU 0 if no argument provided


# Training
if [ "${RETRAIN,,}" = "true" ]; then  # Convert to lowercase for comparison
    echo "Retraining model..."
    
    # Training
    python ../train_model.py \
        --gpu_idx $GPU_IDX \
        --experiment_name $EXPERIMENT_NAME \
        --X_file_postfix $X_FILE_POSTFIX \
        --y_file $Y_FILE \
        --vars $VARS \
        --months $months
fi


# Inspection
python ../inspect_model.py \
    --gpu_idx $GPU_IDX \
    --experiment_name $EXPERIMENT_NAME \
    --X_file_postfix $X_FILE_POSTFIX \
    --y_file 'y_raw.pkl' \
    --vars $VARS \
    --months $months

# LRP plots
python ../lrp_plots.py \
    --gpu_idx $GPU_IDX \
    --experiment_name $EXPERIMENT_NAME \
    --X_file_postfix $X_FILE_POSTFIX \
    --y_file $Y_FILE \
    --vars $VARS \
    --months $months