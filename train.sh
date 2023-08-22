CUDA_VISIBLE_DEVICES=4 \
python train.py \
--data_dir data/data/classification_dataset/images_resample \
--train_anno_file label_all_nohardsample.txt \
--batch-size 4 \
--model uniformer_base_IL \
--img_size 32 128 128 \
--crop_size 28 112 112 \
--eval_metric loss \
--lr 1e-4 \
--min-lr 1e-6 \
--warmup-epochs 5 \
--epochs 200 \
--output output/ \
--mixup

