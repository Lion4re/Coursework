#!/usr/bin/env python
"""
Download WikiArt images and write
Caffe ImageData layer files

This script has been forked from
https://github.com/BVLC/caffe/blob/master/examples/finetune_flickr_style/assemble_data.py
"""
import os
import urllib.request
import argparse
import numpy as np
import pandas as pd
from skimage import io
import multiprocessing
import requests

root_dirname = os.path.abspath(os.path.dirname(__file__))

training_dirname = os.path.join(root_dirname, 'data')

if not os.path.exists(training_dirname):
    os.makedirs(training_dirname)

def download_image(args_tuple):
    url, filename = args_tuple
    try:
        if not os.path.exists(filename):
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
        io.imread(filename)  # Test image readability
        return True
    except KeyboardInterrupt:
        raise
    except:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download a subset of the WikiArt style dataset to a directory.')
    parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help="random seed")
    parser.add_argument(
        '-i', '--images', type=int, default= 4005,
        help="number of images to use (-1 for all [default])",
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=-1,
        help="num workers used to download images. -x uses (all - x) cores [-1 default]."
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Read data, shuffle order, and subsample.
    csv_filename = os.path.join(root_dirname, 'wikiart.csv.gz')
    df = pd.read_csv(csv_filename, index_col=0, compression='gzip')
    df = df.iloc[np.random.permutation(df.shape[0])]
    if args.images > 0 and args.images < df.shape[0]:
        df = df.iloc[:args.images]

    # Make directory for images and get local filenames.
    if training_dirname is None:
        training_dirname = os.path.join(root_dirname, 'data')
    images_dirname = os.path.join(training_dirname, 'images')
    if not os.path.exists(images_dirname):
        os.makedirs(images_dirname)
    df['image_filename'] = [
        os.path.join(images_dirname, _ + '.jpg') for _ in df.index.values
    ]

    # Download images.
    num_workers = args.workers
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count() + num_workers
    print(f'Downloading {args.images} images with {num_workers} workers...')
    pool = multiprocessing.Pool(processes=num_workers)
    map_args = zip(df['image_url'], df['image_filename'])
    results = pool.map(download_image, map_args)

    # Only keep rows with valid images, and write out training file lists.
    df = df[results]
    for split in ['train', 'test']:
        split_df = df[df['_split'] == split]
        filename = os.path.join(training_dirname, '{}.txt'.format(split))
        split_df[['image_filename', 'label']].to_csv(
            filename, sep=' ', header=None, index=None)
    print('Writing train/val for {} successfully downloaded images.'.format(
        df.shape[0]))
