import logging
from tqdm import tqdm
from .util import get_Adience_image_paths, get_nearest_number, remove_nones_Adience
import jsonpickle
import requests
import pickle
import docker
import time
import numpy as np
from collections import Counter
import re
import json


def extract_Adience_arcface(image_type='aligned', docker_port=10002):
    logging.info(f"starting a docker container ...")
    docker_client = docker.from_env()
    arcface_server = docker_client.containers.run(
        "face-analysis", ports={'10002/tcp': docker_port}, detach=True, auto_remove=True)
    logging.debug(f"warming up the container ...")
    time.sleep(10)
    logging.info(f"extracting Adience arcface embedding vectors ...")

    image_paths, folds, header, ages, genders, fold_from, logs = get_Adience_image_paths(
        image_type)
    for image_path in tqdm(image_paths):
        with open(image_path, 'rb') as stream:
            frame_bytestring = stream.read()
        data = {'image': frame_bytestring}
        data = jsonpickle.encode(data)
        response = requests.post(f'http://127.0.0.1:{docker_port}/', json=data)
        response = jsonpickle.decode(response.text)
        fa_results = response['fa_results']

        # with open(image_path + '.pkl', 'wb') as stream:
        #     pickle.dump(fa_results, stream)

        del frame_bytestring, data, response, fa_results

    logging.debug(f"killing the container ...")
    arcface_server.kill()
    logging.info(f"container killed.")
    logging.info(f"DONE!")


def get_Adience_clean(image_type='aligned'):
    image_paths, folds, header, ages, genders, fold_from, logs = get_Adience_image_paths(
        image_type)

    ages = [[int(num) for num in re.findall(r'\d+', age)] for age in ages]
    ages = [np.mean(age) if len(age) else None for age in ages]

    ages = [get_nearest_number(age) for age in ages]
    genders = [gender if gender in ['m', 'f'] else None for gender in genders]

    image_paths, ages, genders, fa_paths, fold_from, embeddings, logs = remove_nones_Adience(
        image_paths, ages, genders, fold_from, logs)

    logs['genders'] = dict(Counter(genders))
    logs['ages'] = dict(Counter(ages))

    with open(f'./data/Adience/meta-data-{image_type}.json', 'w') as stream:
        json.dump(logs, stream, indent=4)

    data = {i: [] for i in range(len(set(fold_from)))}

    assert len(image_paths) == len(ages) == len(
        genders) == len(fold_from) == len(embeddings)

    logging.debug(f"saving data ...")
    for ip, ag, ge, ff, emb in tqdm(zip(image_paths, ages, genders, fold_from, embeddings)):
        data_sample = {'image_path': ip,
                       'age': ag,
                       'gender': ge,
                       'fold': ff,
                       'embedding': emb['normed_embedding']}
        data[ff].append(data_sample)

    np.save(f"./data/Adience/data-{image_type}.npy", data)

    logging.info(f"DONE!")