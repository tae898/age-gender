import json
import logging
import os
import pickle
import re
import time
from collections import Counter
from glob import glob
from pathlib import Path

import jsonpickle
import numpy as np
import pandas as pd
import requests
from numpy.core.numeric import full
from PIL import Image
from python_on_whales import docker
from tqdm import tqdm

from .util import (get_Adience_image_paths, get_meta, get_nearest_number,
                   read_pickle, remove_nones_Adience, resize_square_image,
                   write_json)


def extract_Adience_arcface(
    image_type: str = "aligned",
    docker_port: int = 10002,
    cuda: bool = False,
    resize: int = 640,
):

    if cuda:
        image_name = "tae898/face-detection-recognition-cuda"
        gpus = "all"
    else:
        image_name = "tae898/face-detection-recognition"
        gpus = None

    container = docker.run(
        image=image_name, gpus=gpus, detach=True, publish=[(docker_port, 10002)]
    )

    logging.info(f"starting a docker container ...")
    logging.debug(f"warming up the container ...")
    time.sleep(10)
    logging.info(f"extracting Adience arcface embedding vectors ...")

    (
        image_paths,
        folds,
        header,
        ages,
        genders,
        fold_from,
        logs,
    ) = get_Adience_image_paths(image_type)
    COUNT_FAIL = 0
    for image_path in tqdm(image_paths):
        try:
            if resize:
                image = Image.open(image_path)
                image = resize_square_image(
                    image, width=resize, background_color=(0, 0, 0)
                )

                assert image is not None

                image.save(image_path + ".RESIZED.jpg")
                with open(image_path + ".RESIZED.jpg", "rb") as stream:
                    frame_bytestring = stream.read()
            else:
                with open(image_path, "rb") as stream:
                    frame_bytestring = stream.read()

            data = {"image": frame_bytestring}
            data = jsonpickle.encode(data)
            response = requests.post(f"http://127.0.0.1:{docker_port}/", json=data)
            response = jsonpickle.decode(response.text)
            face_detection_recognition = response["face_detection_recognition"]

            assert face_detection_recognition is not None

            with open(image_path + ".pkl", "wb") as stream:
                pickle.dump(face_detection_recognition, stream)

            del frame_bytestring, data, response, face_detection_recognition
        except Exception as e:
            logging.error(f"failed to process {image_path}: {e}")
            COUNT_FAIL += 1

    logging.error(
        f"in total {COUNT_FAIL} number of images failed to extract face embeddings!"
    )
    logging.debug(f"killing the container ...")
    container.kill()
    logging.info(f"container killed.")
    logging.info(f"DONE!")


def get_Adience_clean(
    image_type: str = "aligned", resize: int = 640, det_score: float = 0.90
):

    (
        image_paths,
        folds,
        header,
        ages,
        genders,
        fold_from,
        logs,
    ) = get_Adience_image_paths(image_type, resize=resize)

    ages = [[int(num) for num in re.findall(r"\d+", age)] for age in ages]
    ages = [np.mean(age) if len(age) else None for age in ages]

    ages = [get_nearest_number(age) for age in ages]
    genders = [gender if gender in ["m", "f"] else None for gender in genders]

    (
        image_paths,
        ages,
        genders,
        fdr_paths,
        fold_from,
        embeddings,
        logs,
    ) = remove_nones_Adience(
        image_paths, ages, genders, fold_from, logs, det_score=det_score
    )

    logs["genders"] = dict(Counter(genders))
    logs["ages"] = dict(Counter(ages))
    logs["folds"] = [len(fold) for idx, fold in enumerate(folds)]

    with open(f"./data/Adience/meta-data-{image_type}.json", "w") as stream:
        json.dump(logs, stream, indent=4)

    data = {i: [] for i in range(len(set(fold_from)))}

    assert (
        len(image_paths)
        == len(ages)
        == len(genders)
        == len(fold_from)
        == len(embeddings)
    )

    logging.debug(f"saving data ...")
    for ip, ag, ge, ff, emb in tqdm(
        zip(image_paths, ages, genders, fold_from, embeddings)
    ):
        data_sample = {
            "image_path": ip,
            "age": ag,
            "gender": ge,
            "fold": ff,
            "embedding": emb["normed_embedding"],
        }
        data[ff].append(data_sample)

    np.save(f"./data/Adience/data-{image_type}.npy", data)

    logging.info(f"DONE!")


def extract_imdb_wiki_arcface(
    dataset: str = "imdb",
    docker_port: int = 10002,
    cuda: bool = False,
    resize: int = 640,
):
    if cuda:
        image_name = "tae898/face-detection-recognition-cuda"
        gpus = "all"
    else:
        image_name = "tae898/face-detection-recognition"
        gpus = None

    container = docker.run(
        image=image_name, gpus=gpus, detach=True, publish=[(docker_port, 10002)]
    )

    logging.info(f"starting a docker container ...")
    logging.debug(f"warming up the container ...")
    time.sleep(10)
    logging.info(f"extracting Adience arcface embedding vectors ...")

    image_paths = glob(f"./data/{dataset}_crop/*/*.jpg")
    COUNT_FAIL = 0
    for image_path in tqdm(image_paths):
        try:
            if resize:
                image = Image.open(image_path)
                image = resize_square_image(
                    image, width=resize, background_color=(0, 0, 0)
                )

                assert image is not None

                image.save(image_path + ".RESIZED.jpg")
                with open(image_path + ".RESIZED.jpg", "rb") as stream:
                    frame_bytestring = stream.read()
            else:
                with open(image_path, "rb") as stream:
                    frame_bytestring = stream.read()

            data = {"image": frame_bytestring}
            data = jsonpickle.encode(data)
            response = requests.post(f"http://127.0.0.1:{docker_port}/", json=data)
            response = jsonpickle.decode(response.text)
            face_detection_recognition = response["face_detection_recognition"]

            assert face_detection_recognition is not None

            with open(image_path + ".pkl", "wb") as stream:
                pickle.dump(face_detection_recognition, stream)

            del frame_bytestring, data, response, face_detection_recognition
        except Exception as e:
            logging.error(f"failed to process {image_path}: {e}")
            COUNT_FAIL += 1

    logging.error(
        f"in total {COUNT_FAIL} number of images failed to extract face embeddings!"
    )
    logging.debug(f"killing the container ...")
    container.kill()
    logging.info(f"container killed.")
    logging.info(f"DONE!")


def get_imdb_wiki_clean(dataset: str, resize: int = 640, det_score: float = 0.9):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/create_db.py
    """
    logging.debug(f"Getting clean data from {dataset} ...")
    root_dir = Path("./")
    data_dir = root_dir.joinpath("data", f"{dataset}_crop")
    mat_path = data_dir.joinpath(f"{dataset}.mat")

    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(
        mat_path, dataset
    )

    genders = []
    ages = []
    img_paths = []
    fdr_paths = []
    sample_num = len(face_score)

    metadata = {"total_num_images": len(full_path)}
    metadata["removed"] = {}
    metadata["removed"]["age_not_correct"] = 0
    metadata["removed"]["gender_not_correct"] = 0
    metadata["removed"]["image_not_correct"] = 0
    metadata["removed"]["no_face_detected"] = 0
    metadata["removed"]["more_than_one_face"] = 0
    metadata["removed"]["bad_quality"] = 0
    metadata["removed"]["no_embeddings"] = 0

    logging.debug(f"Extracting metadata from {dataset} ...")
    for i in tqdm(range(sample_num)):
        if ~(0 <= age[i] <= 100):
            metadata["removed"]["age_not_correct"] += 1
            continue

        if np.isnan(gender[i]):
            metadata["removed"]["gender_not_correct"] += 1
            continue

        img_path = str(data_dir / full_path[i][0])
        fdr_path = img_path + ".pkl"
        if not os.path.isfile(fdr_path):
            metadata["removed"]["image_not_correct"] += 1
            continue

        fdr = read_pickle(fdr_path)

        if fdr is None:
            metadata["removed"]["no_embeddings"] += 1
            continue

        if len(fdr) == 0:
            metadata["removed"]["no_face_detected"] += 1
            continue

        if len(fdr) > 1:
            metadata["removed"]["more_than_one_face"] += 1
            continue

        if fdr[0]["det_score"] < det_score:
            metadata["removed"]["bad_quality"] += 1
            continue

        genders.append({0: "f", 1: "m"}[int(gender[i])])
        ages.append(int(age[i]))
        img_paths.append(img_path)
        fdr_paths.append(fdr_path)

    assert len(genders) == len(ages) == len(img_paths) == len(fdr_paths)

    # outputs = dict(genders=genders, ages=ages, img_paths=img_paths)
    # output_path = data_dir.joinpath(f"{dataset}.csv")
    # df = pd.DataFrame(data=outputs)

    data = []
    logging.debug(f"Saving {dataset} embeddings ...")
    for gender, age, img_path, fdr_path in tqdm(
        zip(genders, ages, img_paths, fdr_paths)
    ):
        fdr = read_pickle(fdr_path)
        assert len(fdr) == 1
        data_sample = {
            "image_path": img_path,
            "age": age,
            "gender": gender,
            "embedding": fdr[0]["normed_embedding"],
        }
        data.append(data_sample)

    metadata["genders"] = dict(Counter(genders))
    metadata["ages"] = dict(Counter(ages))

    logging.info(f"{dataset}'s metadata: {metadata}")
    metadata_write_path = str(data_dir / "meta-data.json")
    write_json(metadata, metadata_write_path)
    logging.info(f"{dataset}'s metadata written at : {metadata_write_path}")

    data_write_path = str(data_dir / "data.npy")
    np.save(data_write_path, data)
    logging.info(f"{dataset}'s data (embeddings) written at : {data_write_path}")

    logging.info(f"DONE!")
