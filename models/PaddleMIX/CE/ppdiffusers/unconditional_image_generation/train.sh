python -u -m paddle.distributed.launch --gpus "0,1,2,3"  train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --cache_dir 'data' \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-flowers-64" \
  --train_batch_size=16 \
  --num_epochs=1 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500