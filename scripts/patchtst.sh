seq_len=800000
pred_len=5000
patch_size=1000
stride=1000
model=patchtst
batch_size=32
epochs=2
lr=1e-3
itr=1
task=classification

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
    --loss cross_entropy \
    --key_metric mcc \
    --seed 2025 \
    --batch_size $batch_size \
    --lr $lr \
    --itr $itr \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --d_model 128 \
    --n_heads 6 \
    --d_ff 512 \
    --dropout 0.1 \
    --enc_in 3 \
    --patch_size $patch_size \
    --stride $stride \
    --n_layer 6 \
    --patience 30 \
