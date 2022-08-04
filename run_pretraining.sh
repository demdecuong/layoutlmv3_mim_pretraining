# Set the path to save checkpoints
OUTPUT_DIR=/home/minh/work/fpt/RESUME/pretraining/beit/output
DATA_PATH=../data/
TOKENIZER_PATH=../dall_e_tokenizer_weight


# python run_beit_pretraining.py \
python run_layout_pretraining.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 16 --lr 1e-4 --warmup_steps 100 --epochs 300 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1