import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
import sys

np.random.seed(0)
random.seed(0)

def generate_data_description(save_dir, img_dir, dataset_name):

    dataset = EasyDict()
    dataset.description = dataset_name
    dataset.root = img_dir
    
    image_name = os.listdir(img_dir)
    img_len = len(image_name)
    print(img_len)
    
    dataset.image_name = image_name
    dataset.partition = np.arange(0, img_len)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


dataset_name = 'tinycrop'
data_dir = sys.argv[1]

if dataset_name == 'inaturalist':
    save_dir = f'{data_dir}/inaturalist/'
    img_dir = f'{data_dir}/inaturalist/iNaturalist/images'
elif dataset_name == 'isun':
    save_dir = f'{data_dir}/isun/'
    img_dir = f'{data_dir}/isun/iSUN/iSUN_patches'
elif dataset_name == 'lsuncrop':
    save_dir = f'{data_dir}/lsuncrop/'
    img_dir = f'{data_dir}/lsuncrop/LSUN/test'
elif dataset_name == 'lsunre':
    save_dir = f'{data_dir}/lsunre/'
    img_dir = f'{data_dir}/lsunre/LSUN_resize/LSUN_resize/'
elif dataset_name == 'places':
    save_dir = f'{data_dir}/places/'
    img_dir = f'{data_dir}/places/Places/images'
elif dataset_name == 'sun':
    save_dir = f'{data_dir}/sun/'
    img_dir = f'{data_dir}/sun/SUN/images'
elif dataset_name == 'texture':
    save_dir = f'{data_dir}/texture'
    img_dir = f'{data_dir}/texture/images/'
elif dataset_name == 'tinycrop':
    save_dir = f'{data_dir}/tinycrop'
    img_dir = f'{data_dir}/tinycrop/test/'
elif dataset_name == 'tinyre':
    save_dir = f'{data_dir}/tinyre'
    img_dir = f'{data_dir}/tinyre/Imagenet_resize/Imagenet_resize'
    
generate_data_description(save_dir,img_dir,dataset_name)
