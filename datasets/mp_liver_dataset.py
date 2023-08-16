import torch
import numpy as np
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from ignite.distributed import DistributedProxySampler
from scipy import ndimage, signal
import albumentations as A
import sys
import os
import cv2
try:
    from datasets.transforms import *
except:
    from transforms import *
from torch.utils.data import BatchSampler

class MultiPhaseLiverDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_training=True, is_cls2=False):
        self.args = args
        self.size = args.img_size
        self.is_training = is_training
        img_list = []
        lab_list = []
        phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase', 
                      'C-pre', 'C+A', 'C+V', 'C+Delay']
        self.benign_cls = [0, 2, 4, 5]
        self.mag_cls = [1, 3, 6]
        self.is_cls2 = is_cls2

        if is_training:
            anno = np.loadtxt(args.train_anno_file, dtype=np.str_)
        else:
            anno = np.loadtxt(args.val_anno_file, dtype=np.str_)

        for item in anno:
            mp_img_list = []
            # mp_imgopt_list = []
            for phase in phase_list:
                # if is_training:
                    mp_img_list.append(f'{args.data_dir}/{item[0]}/{phase}.nii.gz')
                # else:
                    # mp_img_list.append(f'{args.val_data_dir}/{item[0]}/{phase}.nii.gz')
                # mp_imgopt_list.append(f'{args.data_dir}_opt/{item[0]}/{phase}.npy')
            img_list.append(mp_img_list)
            # imgopt_list.append(mp_imgopt_list)
            lab_list.append(item[1])

        self.img_list = img_list
        # self.imgopt_list = imgopt_list
        self.lab_list = lab_list

    def __getitem__(self, index):
        self.seed = random.random()
        args = self.args
        image = self.load_mp_images(self.img_list[index])
        label = int(self.lab_list[index])
        if self.is_training:
            if args.mixup:
                if label == 6:
                    image = self.mixup(image, label, seed=1) 
                else:
                    image = self.mixup(image, label)
                image = image.copy()

            image = self.transforms(image, args.train_transform_list)
            seed = random.random()
            if seed > 0.5:
                image = self.get_mask(image)
            else:
                image = self.get_hidden(image)

        else:
            image = self.transforms(image, args.val_transform_list)
        image = image.copy()
            
        img_id = self.img_list[index][0].split('/')[-2]

        if self.is_cls2:
            if label in self.benign_cls:
                label_cls2 = 0
            else:
                label_cls2 = 1
            if self.is_training:
                return (image, label, label_cls2)
            else:
                return (image, label, img_id, label_cls2)
        else:
            if self.is_training:
                return (image, label)
            else:
                return (image, label, img_id)
    
    def get_mask(self, image):
        seed = random.random()
        if seed > 0.9:
            image = Grid_Mask(image)
        return image
    
    def get_hidden(self, image):
        seed = random.random()
        if seed > 0.9:
            image = hide_patch(image)
        return image

    def mixup(self, image, label, seed=1.0):
        if seed > 0.7:
            alpha = 1.0
            all_target_ind = [i for i, x in enumerate(self.lab_list) if int(x) == label]
            index = random.choice(all_target_ind)
            image_bal = self.load_mp_images(self.img_list[index])
            lam = np.random.beta(alpha, alpha)
            image = lam * image + (1 - lam) * image_bal

        return image

    def load_mp_images(self, mp_img_list):
        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            image = resize3D(image, self.size)
            image = image_normalization(image)
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image

    def transforms(self, mp_image, transform_list):
        args = self.args
        if 'center_crop' in transform_list:
            mp_image = center_crop(mp_image, args.crop_size)
        if 'random_crop' in transform_list:
            mp_image = random_crop(mp_image, args.crop_size)
        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        # if self.is_training:
        #     strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
        #     index = random.randrange(len(strategy))
        #     employ = strategy.pop(index)
        #     mp_image = pixel_aug(mp_image.astype(np.float32), employ[0])
        #     mp_image = spatial_aug(mp_image, employ[1])
        # return mp_image.astype(np.float32)
        if self.seed > 0.95:
            if 'edge' in transform_list:
                mp_image = edge(mp_image)
        elif self.seed > 0.9:
            if 'emboss' in transform_list:
                mp_image = emboss(mp_image)
        elif self.seed > 0.7:
            if 'filter' in transform_list:
                seed2=random.random()
                if seed2>0.8:
                    mp_image = blur(mp_image)
                elif seed2>0.6:
                    mp_image = sharpen(mp_image)
                elif seed2>0.3:
                    mp_image = mask(mp_image)
        return mp_image.astype(np.float32)

    def __len__(self):
        return len(self.img_list)


def collate_fn_eval_train(data):
    assert type(data) != dict
    res_data = {}
    batch_size = len(data)
    D, C, H, W = data[0][0].shape
    res_data['img_path'] = [eval_data[2] for eval_data in data]
    res_data['img'] = torch.cat([torch.from_numpy(eval_data[0]) for eval_data in data]).reshape(-1, D, C, H, W)
    res_data['label'] = torch.cat([torch.tensor(eval_data[1]).unsqueeze(0) for eval_data in data])
    if len(data[0]) == 4:
        res_data['label_twocls'] = torch.cat([torch.tensor(eval_data[3]).unsqueeze(0) for eval_data in data])
    return res_data

def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
        mode='instance'
):
    weights = get_sampling_probabilities(dataset,mode=mode)

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            sampler = DistributedProxySampler(
                            ExhaustiveWeightedRandomSampler(weights, num_samples=len(dataset))
                        )
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
        if is_training:
            sampler = torch.utils.data.WeightedRandomSampler(weights=weights,num_samples = len(dataset),replacement=True)


    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader

def create_balanced_sample_dataloader(
        dataset=None,
        is_training=False,
        distributed=False,
        num_workers=1,
        num_samples=10,
        collate_fn=None,
):
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        batch_sampler = DistributedProxySampler(
                        BalancedBatchSampler(dataset, None , len(dataset.mag_cls), num_samples)
                    )
    else:
        batch_sampler = BalancedBatchSampler(dataset, None , len(dataset.mag_cls), num_samples)
    
    loader_args = dict(
        shuffle=False,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        # loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, labels, n_classes, n_samples):
        self.dataset = dataset

        if labels != None:
            self.labels = labels
        else:
            self.labels = dataset.get_labels()
            
            
        self.labels = np.array(self.labels)
        self.labels_set = np.array(list(set(self.labels)))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    # [[same_class*n_samples]*n_classes] --> [[class1 class2 ... classn] * n_samples]
    def _rearange(self,indices):
        index_list = []
        for idx in range(self.n_samples):
            index_list.extend(indices[idx::self.n_samples])
        return index_list

    # [[same_class*n_samples]*n_classes]
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # TODO: 这个classes随机选的几个，感觉不能保证每类都选取到
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            
            for class_ in classes:
                # 最后一轮数量不够了,变成从所有样本中重新选择
                if self.used_label_indices_count[class_]+self.n_samples>len(self.label_to_indices[class_]):
                    for i in range(self.n_samples):
                        chose = np.random.choice(len(self.label_to_indices[class_]), 1 , replace=False)
                        indices.extend(self.label_to_indices[class_][chose])

                # 按顺序选择每个类别样本(但是每个类别的样本列表是打乱的，所以相当于随机)
                else:
                    indices.extend(self.label_to_indices[class_][
                                     self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                
                # 记录未使用的起始index
                self.used_label_indices_count[class_] += self.n_samples

                # 如果该类别的样本快用完了，则重新打乱该类别的样本,并重置未使用的起始index
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            indices = self._rearange(indices)
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

def get_sampling_probabilities(dataset,mode,ep=None, n_eps=None):
    np_lab_list = list(map(int,dataset.lab_list))
    class_count = np.unique(np_lab_list,return_counts=True)[1]
    sampling_probs = count_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[np_lab_list]

    return sample_weights

def count_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

if __name__ == "__main__":
    import yaml
    import parser
    import argparse
    from tqdm import tqdm

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '--data_dir', default='data/classification_dataset/images/', type=str)
    parser.add_argument(
        '--train_anno_file', default='data/classification_dataset/labels/train_fold1.txt', type=str)
    parser.add_argument(
        '--val_anno_file', default='data/classification_dataset/labels/val_fold1.txt', type=str)
    parser.add_argument('--train_transform_list', default=['random_crop',
                                                           'z_flip',
                                                           'x_flip',
                                                           'y_flip',
                                                           'rotation',],
                        nargs='+', type=str)
    parser.add_argument('--val_transform_list',
                        default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--img_size', default=(16, 128, 128),
                        type=int, nargs='+', help='input image size.')
    parser.add_argument('--crop_size', default=(14, 112, 112),
                        type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float,
                        help='Random flip prob (default: 0.5)')
    parser.add_argument('--angle', default=45, type=int)

    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)
        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text
    
    args, args_text = _parse_args()
    args_text = yaml.load(args_text, Loader=yaml.FullLoader)
    args_text['img_size'] = 'xxx'
    print(args_text)

    args.distributed = False
    args.batch_size = 100

    dataset = MultiPhaseLiverDataset(args, is_training=True)
    data_loader = create_loader(dataset, batch_size=3, is_training=True)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    for images, labels in data_loader:
        print(images.shape)
        print(labels)

    # val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    # val_data_loader = create_loader(val_dataset, batch_size=10, is_training=False)
    # for images, labels in val_data_loader:
    #     print(images.shape)
    #     print(labels)