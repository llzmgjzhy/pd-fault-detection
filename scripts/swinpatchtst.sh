seq_len=800000
patch_size=500
stride=500
window_num=32
model=swinPatchtst
batch_size=32
epochs=100
lr=5e-4
itr=1
task=fault_detection

python main_patchtst.py \
    --task $task \
    --comment "$task using $model" \
    --details "fault detection instead of classification." \
    --name "${task}_vsb" \
    --root_path ./dataset \
    --meta_path vsb-power-line-fault-detection \
    --data_path three-phase-denoise-features \
    --output_dir ./experiments \
    --records_file vsb_$task.xlsx \
    --model_name $model \
    --epochs $epochs \
    --loss bce \
    --key_metric loss \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --seq_len $seq_len \
    --d_model 128 \
    --n_heads 4 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0.2 \
    --enc_in 3 \
    --patch_size $patch_size \
    --stride $stride \
    --n_layer 6 \
    --window_num $window_num \
    --patience 100 \
    # --weight_decay 1e-3 \
