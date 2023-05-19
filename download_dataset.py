'https://pjreddie.com/projects/pascal-voc-dataset-mirror/'

import os
import random
from pathlib import Path

import tarfile
import urllib.request

def move_files(train_dir, test_dir, validation_size=5000):
    train_images = os.listdir(train_dir / 'JPEGImages')
    test_images = random.sample(train_images, validation_size)

    for path in test_images:
        img_name = path.split('/')[-1].split('.')[0]   # path/to/image.jpg -> image.jpg -> image

        train_image = train_dir / 'JPEGImages' / f'{img_name}.jpg'
        test_image = test_dir / 'JPEGImages' / f'{img_name}.jpg'

        train_annotation = train_dir / 'Annotations' / f'{img_name}.xml'
        test_annotation = test_dir / 'Annotations' / f'{img_name}.xml'

        os.rename(train_image, test_image)            # move image
        os.rename(train_annotation, test_annotation)  # move annotation

def request_tar_file(url, extract_dir, file_name):
    urllib.request.urlretrieve(url, file_name)

    print('[*] Extracting tar file...')
    tar = tarfile.open(file_name, "r:")
    tar.extractall(extract_dir)
    tar.close()
    print('[*] Done.\n')

    os.remove(file_name)


def download_dataset(dataset, train_dir, test_dir, train_data_link, test_data_link=None, validation_size=5000):
    print(f'[*] Downloading {dataset} train dataset...')
    print(f'[*] url from: {train_data_link}')
    request_tar_file(url=train_data_link, extract_dir=train_dir, file_name='voctrain.tar')

    if test_data_link is not None:
        print(f'[*] Downloading {dataset} validation dataset...')
        print(f'[*] url from: {test_data_link}')
        request_tar_file(url=test_data_link, extract_dir=test_dir, file_name='voctest.tar')

    else:
        print('[*] Moving validation data from train dataset...')
        train_dir = train_dir / 'VOCdevkit' / dataset
        test_dir = test_dir / 'VOCdevkit' / dataset

        os.makedirs(test_dir / 'Annotations', exist_ok=True)
        os.makedirs(test_dir / 'JPEGImages', exist_ok=True)

        move_files(train_dir, test_dir, validation_size)
        print('[*] Done.\n')

def download_VOC2007():
    train_dir = Path('./dataset/pascalVOC/content/VOCtrain/')
    test_dir = Path('./dataset/pascalVOC/content/VOCtest/')
    train_data_link = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
    test_data_link = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'

    download_dataset('VOC2007', train_dir, test_dir, train_data_link, test_data_link)

def download_VOC2012():
    train_dir = Path('./dataset/pascalVOC/content/VOCtrain/')
    test_dir = Path('./dataset/pascalVOC/content/VOCtest/')
    train_data_link = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    test_data_link = None

    download_dataset('VOC2012', train_dir, test_dir, train_data_link, test_data_link)

if __name__ == '__main__':
    # download_VOC2007()
    download_VOC2012()