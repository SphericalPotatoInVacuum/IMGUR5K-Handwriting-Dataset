import multiprocessing as mp
from time import sleep
from loguru import logger
import hashlib
import json
import numpy as np
import os
import requests
import pandas as pd


# Image hash computed for image using md5..
def compute_image_hash(img_path):
    return hashlib.md5(open(img_path, 'rb').read()).hexdigest()


# Create a sub json based on split idx
def _create_split_json(anno_json, _split_idx):

    split_json = {}

    split_json['index_id'] = {}
    split_json['index_to_ann_map'] = {}
    split_json['ann_id'] = {}

    for _idx in _split_idx:
        # Check if the idx is not bad
        if _idx not in anno_json['index_id']:
            continue

        split_json['index_id'][_idx] = anno_json['index_id'][_idx]
        split_json['index_to_ann_map'][_idx] = anno_json['index_to_ann_map'][_idx]

        for ann_id in split_json['index_to_ann_map'][_idx]:
            split_json['ann_id'][ann_id] = anno_json['ann_id'][ann_id]

    return split_json


def process_index(index, img_hash, invalid_urls, tot_evals, num_match):
    if os.path.exists(f'images/{index}.jpg') and img_hash == compute_image_hash(f'images/{index}.jpg'):
        # logger.info(f'Image {index} already exists, skipping')
        num_match.value += 1
        tot_evals.value += 1
        return

    image_url = f'https://i.imgur.com/{index}.jpg'
    success = False
    while not success:
        try:
            logger.info(f'Fetching image {index}')
            img_data = requests.get(image_url).content
            success = True
        except requests.exceptions.ConnectionError:
            logger.warning(f'Internet problems on image {index}, trying again in 1 second...')
            sleep(1.0)
    if len(img_data) < 100:
        logger.error(f"URL retrieval for {index} failed!!\n")
        invalid_urls.append(image_url)

    with open(f'images/{index}.jpg', 'wb') as handler:
        handler.write(img_data)

    tot_evals.value += 1
    if img_hash != compute_image_hash(f'images/{index}.jpg'):
        logger.error(f"For IMG: {index}, ref hash: {img_hash} != cur hash: {compute_image_hash(f'images/{index}.jpg')}")
        os.remove(f'images/{index}.jpg')
        invalid_urls.append(image_url)
        return
    logger.success(f'Fetched image {index}')
    num_match.value += 1


def main():
    os.makedirs('images', exist_ok=True)

    # Create a hash dictionary with image index and its correspond gt hash
    with open(f"dataset_info/imgur5k_hashes.lst", "r", encoding="utf-8") as _H:
        hashes = _H.readlines()
        hash_dict = {}

        for hash in hashes:
            hash_dict[f"{hash.split()[0]}"] = f"{hash.split()[1]}"

    # Download the urls and save only the ones with valid hash to ensure underlying image has not changed
    tot_evals = 0
    num_match = 0
    invalid_urls = []
    manager: mp.Manager

    with mp.Pool(16) as pool, mp.Manager() as manager:
        invalid_urls_sh = manager.list()
        tot_evals_sh = manager.Value('d', 0)
        num_match_sh = manager.Value('d', 0)

        pool.starmap(process_index, [(index, img_hash, invalid_urls_sh, tot_evals_sh, num_match_sh)
                     for index, img_hash in hash_dict.items()], chunksize=16)

        tot_evals = tot_evals_sh.value
        num_match = num_match_sh.value
        invalid_urls = invalid_urls_sh[:]

    logger.success(f"MATCHES: {num_match}/{tot_evals}")
    # Generate the final annotations file
    # Format: { "index_id" : {indexes}, "index_to_annotation_map" : { annotations ids for an index}, "annotation_id": { each annotation's info } }
    # Bounding boxes with '.' mean the annotations were not done for various reasons

    _F = pd.read_csv('dataset_info/imgur5k_data.lst', sep='\t', encoding='utf-8', engine='python', quoting=3, dtype=str)
    _F = _F.to_numpy()
    anno_json = {}

    anno_json['index_id'] = {}
    anno_json['index_to_ann_map'] = {}
    anno_json['ann_id'] = {}

    cur_index = ''
    for cnt, image_url in enumerate(_F[:, 0]):
        if image_url in invalid_urls:
            continue

        index = image_url.split('/')[-1][:-4]
        if index != cur_index:
            anno_json['index_id'][index] = {
                'image_url': image_url,
                'image_path': f'images/{index}.jpg',
                'image_hash': hash_dict[index]}
            anno_json['index_to_ann_map'][index] = []

        ann_id = f"{index}_{len(anno_json['index_to_ann_map'][index])}"
        anno_json['index_to_ann_map'][index].append(ann_id)
        try:
            anno_json['ann_id'][ann_id] = {'word': _F[cnt, 2], 'bounding_box': json.loads(_F[cnt, 1])}
        except BaseException:
            print(cnt, _F[cnt, 1])

        cur_index = index

    json.dump(anno_json, open('dataset_info/imgur5k_annotations.json', 'w'), indent=4)

    # Now split the annotations json in train, validation and test jsons
    splits = ['train', 'val', 'test']
    for split in splits:
        _split_idx = np.loadtxt(f'dataset_info/{split}_index_ids.lst', delimiter="\n", dtype=str)
        split_json = _create_split_json(anno_json, _split_idx)
        json.dump(split_json, open(f'dataset_info/imgur5k_annotations_{split}.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
