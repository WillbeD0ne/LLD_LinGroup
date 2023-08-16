import random
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy import ndimage, signal
from PIL import Image
from PIL import ImageFilter
from timm.models.layers import to_3tuple, to_2tuple

import albumentations as A
import math
import random
import cv2

from grid_mask import GridMask

def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array

def resize2D(image, size):
    size = to_2tuple(size)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(image, size=size, mode='trilinear', align_corners=True).squeeze(0).squeeze(0)
    return x.cpu().numpy()

def resize3D(image, size):
    """
    重采样 指定spacing
    """
    size = to_3tuple(size)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(image, size=size, mode='trilinear', align_corners=True).squeeze(0).squeeze(0)
    return x.cpu().numpy()

def image_normalization_2D(image, win=None, adaptive=True):
    """
    image: list
    """
    min_list = [np.min(img) for img in image]
    min_ = min(min_list)
    max_list = [np.max(img) for img in image]
    max_ = max(max_list)
    img_normalization = []
    for img in image:
        img = (image - min_) / (max_ - min_)
        img_normalization.append(img)
    return img_normalization

def image_normalization(image, win=None, adaptive=True):
    if win is not None:
        image = 1. * (image - win[0]) / (win[1] - win[0])
        image[image < 0] = 0.
        image[image > 1] = 1.
        return image
    elif adaptive:
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)
        return image
    else:
        return image

def random_crop(image, crop_shape):
    crop_shape = to_3tuple(crop_shape)
    _, z_shape, y_shape, x_shape = image.shape
    z_min = np.random.randint(0, z_shape - crop_shape[0])
    y_min = np.random.randint(0, y_shape - crop_shape[1])
    x_min = np.random.randint(0, x_shape - crop_shape[2])
    image = image[..., z_min:z_min+crop_shape[0], y_min:y_min+crop_shape[1], x_min:x_min+crop_shape[2]]
    return image

def random_crop_2D(image_array, crop_shape):
    crop_shape = to_2tuple(crop_shape)
    slice_num, y_shape, x_shape = image_array.shape
    y_min = np.random.randint(0, y_shape - crop_shape[0])
    x_min = np.random.randint(0, x_shape - crop_shape[1])
    image = image_array[..., y_min:y_min+crop_shape[0], x_min:x_min+crop_shape[1]]

    return image

def center_crop(image, target_shape=(10, 80, 80)):
    target_shape = to_3tuple(target_shape)
    b, z_shape, y_shape, x_shape = image.shape
    z_min = z_shape // 2 - target_shape[0] // 2
    y_min = y_shape // 2 - target_shape[1] // 2
    x_min = x_shape // 2 - target_shape[2] // 2
    image = image[:, z_min:z_min+target_shape[0], y_min:y_min+target_shape[1], x_min:x_min+target_shape[2]]
    return image

def center_crop_2D(image_array, target_shape=(80, 80)):
    target_shape = to_2tuple(target_shape)
    z_shape, y_shape, x_shape = image_array.shape
    y_min = y_shape // 2 - target_shape[0] // 2
    x_min = x_shape // 2 - target_shape[1] // 2
    image = image[:, y_min:y_min+target_shape[0], x_min:x_min+target_shape[1]]
    return image

def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, ::-1, ...]

def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[..., ::-1]

def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, :, ::-1, ...]

def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image

def rotate(image, angle=10):
    angle = random.randint(-10, 10)
    # angle=-10
    # print('random rotate={} in rotate in transforms.py'.format(angle))
    r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    if r_image.shape != image.shape:
        r_image = center_crop(r_image, target_shape=image.shape[1:])
    return r_image

class Grid(object):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        d = np.random.randint(self.d1, self.d2)
        #d = self.d
        
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask

        mask = mask.expand_as(img)
        img = img * mask 

        return img


def pixel_aug(image, transform_num):
    num_model, Z, H, W= image.shape
    transforms = A.Compose([
            A.ColorJitter(brightness=3, contrast=0, saturation=0, hue=0, p=1),
            A.ColorJitter(brightness=0, contrast=3.0, saturation=0, hue=0, p=1),
            # A.Posterize(num_bits=math.floor(4.0), p=1),
            A.Sharpen(alpha=(0.4, 0.8), lightness=(1, 1), p=1),
            A.GaussianBlur(blur_limit=(1, 9), p=1),
            A.GaussNoise(var_limit=(0.001, 0.005), mean=0, per_channel=True, p=1),
            # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=1, num_steps=10),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    ])
    img_transforms = random.sample(transforms[0:], transform_num)
    img_transforms = A.Compose([*img_transforms])

    # visualization
    # viss = []
    # for i, tran in enumerate(transforms):
    #     vis = tran(image=image[0, 6, :])['image']
    #     vis *= 255.0/np.max(vis)
    #     viss.append(vis)
    # ori = image[0, 6, :]
    # ori *= 255./np.max(ori)
    # viss.append(ori)
    # viss = np.concatenate(viss, axis=0)
    # cv2.imwrite('test.png', viss)

    image_transformed = []
    for m in range(num_model):
        for z in range(Z):
            img_slice = image[m, z, :]
            image_transformed.append(img_transforms(image=img_slice)['image'])
    image_transformed = np.stack(image_transformed, axis=0).reshape(num_model, Z, H, W)
    return image_transformed

def spatial_aug(image, transform_num):
    num_model, Z, H, W= image.shape
    transforms = A.Compose([
            A.Rotate(limit=20, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=1),
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Affine(scale=(0.8, 1.2), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=1),
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 10), 'y': (0, 0)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=1),  # x
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 0), 'y': (0, 10)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=1),
            A.Affine(scale=None, translate_percent={'x': (0, 0.1), 'y': (0, 0)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=1),
            A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.1)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=1) ])
    img_transforms = random.sample(transforms[0:], transform_num)
    img_transforms = A.Compose([*img_transforms])

    # visualization
    # viss = []
    # for i, tran in enumerate(transforms):
    #     vis = tran(image=image[0, 6, :])['image']
    #     vis *= 255.0/np.max(vis)
    #     viss.append(vis)
    # ori = image[0, 6, :]
    # ori *= 255./np.max(ori)
    # viss.append(ori)
    # viss = np.concatenate(viss, axis=0)
    # cv2.imwrite('test.png', viss)

    image_transformed = []
    for m in range(num_model):
        for z in range(Z):
            img_slice = image[m, z, :]
            image_transformed.append(img_transforms(image=img_slice)['image'])
    image_transformed = np.stack(image_transformed, axis=0).reshape(num_model, Z, H, W)
    return image_transformed

#边缘检测
def edge(image):
    seed1=random.random()
    edge_mode=''
    
    #[0.8,0.1,0.1]
    if seed1>0.2:
        edge_mode='reflect'
    elif seed1>0.1:
        edge_mode='wrap'
    elif seed1>0.:
        edge_mode='constant'

    seed2=random.random()
    #[0.4,0.4,0.2]
    if edge_mode:
        if seed2>0.6:
            image = ndimage.sobel(image, axis=-1,mode=edge_mode)
        elif seed2>0.2:
            image = ndimage.sobel(image, axis=-2,mode=edge_mode)
        elif seed2>0.:
            image = ndimage.sobel(image, axis=-3,mode=edge_mode)
    return image

def blur(image):
    seed1=random.random()
    #[0.2,0.4,0.4]
    if seed1>0.8:
        image = ndimage.gaussian_filter(image, sigma=1)#高斯模糊#[1~1.5]#0.3s/gragh
    elif seed1>0.4:
        image = ndimage.median_filter(image,size=2)#中值滤波模糊#size=3,5s/graph;size=2,1s/gragh
    elif seed1>0.:
        image = signal.wiener(image, mysize=3)#维纳滤波模糊#size=5,2s/graph;size=3,1.6s/gragh
        image=image.astype(np.float32)
    return image



def sharpen(image):
    D, Z, H, W = image.shape
    image_ls=[]
    seed1=random.random()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_seg=Image.fromarray(np.uint8(255*image[i,j,:,:]))
            #[0.3 0.3 0.2 0.2]
            if seed1>0.7:
                img_seg=img_seg.filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif seed1>0.4:
                img_seg=img_seg.filter(ImageFilter.EDGE_ENHANCE)
            elif seed1>0.2:
                img_seg=img_seg.filter(ImageFilter.DETAIL)  
            elif seed1>0.:
                img_seg=img_seg.filter(ImageFilter.SHARPEN)
            image_ls.append(np.asarray(img_seg)[None, ...])
    image_ls=np.concatenate(image_ls, axis=0)
    image=image_ls.reshape(D, Z, H, W)/255.
    image=image.astype(np.float32)
    
    return image

def emboss(image):
    D, Z, H, W = image.shape
    image_ls=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_seg=Image.fromarray(np.uint8(255*image[i,j,:,:]))
            img_seg=img_seg.filter(ImageFilter.EMBOSS)
            image_ls.append(np.asarray(img_seg)[None, ...])
    image_ls=np.concatenate(image_ls, axis=0)
    image=image_ls.reshape(D, Z, H, W)/255.
    # image = torch.Tensor(image)#https://blog.csdn.net/qq_42346574/article/details/120100424
    image=image.astype(np.float32)
    return image

#钝化掩蔽
def mask(image):
    mask1 = ndimage.gaussian_filter(image, sigma=1)#钝化
    mask3 = ndimage.gaussian_filter(image, sigma=3)#钝化
    image = mask3 +6*(mask3-mask1)#钝化掩蔽作差
    return image

def hide_patch(image, hide_prob=0.5):
    num_model, Z, H, W = image.shape
    image_ls=[]
    for i in range(num_model):
        z_grid_sizes = [1, 2]
        grid_sizes = [4, 8, 16]
        # grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]
        grid_size = random.choice(grid_sizes)
        z_grid_size = random.choice(z_grid_sizes)
        for x in range(0, W, grid_size):
            for y in range(0, H, grid_size):
                for z in range(0, Z, z_grid_size):
                    x_end = min(W, x+grid_size)  
                    y_end = min(H, y+grid_size)
                    z_end = min(Z, z+z_grid_size)
                    if random.random() <= hide_prob:
                        noice_matrix = np.random.normal(0, 0.5, size=[z_end-z, y_end-y, x_end-x]).astype(np.float32) 
                        image[i, z:z_end, y:y_end, x:x_end] = noice_matrix
        
    # image_ls=np.concatenate(image_ls, axis=0)
    # image=image_ls.reshape(8,14,112,112)/255.
    image=image.astype(np.float32)
    return image

def Grid_Mask(image):
    gridmask = GridMask(mode=0)
    num_model, Z, H, W= image.shape
    image_ls=[]
    for i in range(num_model):
        z_len = random.randint(1, Z)
        z_start = random.randint(0, Z-1)
        z_end = min(Z, z_start+z_len)
        image_block = image[i, z_start:z_end, :]
        image_block = gridmask(image_block)
        image[i, z_start:z_end, :] = image_block
    image=image.astype(np.float32)
    return image
