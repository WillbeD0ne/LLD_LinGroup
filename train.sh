# CUDA_VISIBLE_DEVICES=4 \
# python train.py \
# --data_dir data/data/classification_dataset/images_resample \
# --val_data_dir data_test/lldmmri_test_set/classification_dataset/images_resample \
# --train_anno_file data/data/classification_dataset/labels/label_all_nohardsample.txt \
# --val_anno_file data/data/classification_dataset/labels/labels_test_custom.txt \
# --batch-size 4 \
# --model uniformer_small_IL \
# --img_size 32 128 128 \
# --crop_size 28 112 112 \
# --lr 1e-4 \
# --min-lr 1e-6 \
# --warmup-epochs 5 \
# --epochs 200 \
# --output output/unpretrained_sample_aug/7cls_test_resample_base/ \
# --mixup

