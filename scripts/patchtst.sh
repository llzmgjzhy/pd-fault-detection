seq_len=800000
pred_len=5000
patch_size=1000
stride=1000
model=patchtst
batch_size=32
epochs=100
lr=5e-4
itr=1
task=fault_detection

python main_patchtst.py \
    --task $task \
    --comment "$task using $model" \
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
    --pred_len $pred_len \
    --d_model 128 \
    --n_heads 4 \
    --d_ff 256 \
    --dropout 0.2 \
    --enc_in 3 \
    --patch_size $patch_size \
    --stride $stride \
    --n_layer 6 \
    --patience 100 \
