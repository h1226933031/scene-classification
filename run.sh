export CUDA_VISIBLE_DEVICES=1
python -u run.py \
--model_name 'CNN' \
--data_augmentation True \
--epochs 100 \
--batch_size 128 \
--lr 0.005