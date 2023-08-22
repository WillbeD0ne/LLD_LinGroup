# LLD-MMRI2023_LinGroup
The code of LLD-MMRI2023 competition:
*https://github.com/LMMMEng/LLD-MMRI2023*

Competition team: LinGroup

## Data Preparation
We first resample all MRI ori data (including train set, val set and test set) and place it on a new fold.

The command is following:
```
python preprocess/crop_roi.py --data-dir {your data path} --anno-path {Annotation path} --save-dir {your save path}
```
Thus we obtain the resampled data of train set, val set and test set. 

## Uniformer pretrained model Preparation
In order to alleviate the preblem of overfitting, we use the pretrained model of Uniformer which trained on K400 dataset for video action classification task.

The pretrained model for uniformer can be found in:
*https://github.com/Sense-X/UniFormer/tree/main/video_classification*

We choose model `uniformer_base_k400_16x4` weight and save it on fold [pretrain_model](https://github.com/WillbeD0ne/LLD_LinGroup/pretrain_model)


## Train
Through training and validating on cross-fold, we select almost 10 cases of MRI data as the hard samples. Then these hard samples are removed from the combination of train set and val set. The novel label file is written on `label_all_nohardsample.txt`.

During the training process, we save model weights every 10 epochs between 80 and 200. And we regard the mdoel with lowest loss as the best model.

The command is following:
```
bash train.sh
```
or
```
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir {resampled data path} --train_anno_file label_all_nohardsample.txt --batch-size 4 --model uniformer_base_IL --img_size 32 128 128 --crop_size 28 112 112 eval_metric loss --min-lr 1e-6 --warmup-epochs 5 --epochs 200 --output output --mixup
```

## Checkpoint weight
You can download the trained model weight on [Goole Drive](https://drive.google.com/file/d/1VQOSL0OWuZ6yv5lkIxQy5DmVMr3xhS_3/view?usp=sharing)

## Evaluation
After downloading the checkpoint, you can evaluate the model by using the following command: 
```
python predict.py --val_data_dir {your test set path} --checkpoint {checkpoint path} --batch-size 8 --results-dir {result path} --team_name LinGroup
```
And the result of json file will be saved on the result path you create. 

