import json
import logging
import os
import pickle
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def read_pickle(pickle_path: str):
    with open(pickle_path, "rb") as stream:
        foo = pickle.load(stream)
    return foo


def update_lr_scheduler(config: dict, train_dataloader_size: int):
    """
    Update the key values in the dict so that the lr_scheduler is compatible
    with the optimizer and trainer

    config: ConfigParser object
    """
    if config["lr_scheduler"]["type"] == "OneCycleLR":
        config["lr_scheduler"]["args"].update(
            {"max_lr": config["optimizer"]["args"]["lr"]}
        )

        config["lr_scheduler"]["args"].update({"epochs": config["trainer"]["epochs"]})

        config["lr_scheduler"]["args"].update(
            {"steps_per_epoch": train_dataloader_size}
        )

    return config


def expand2square(pil_img: Image, background_color: tuple):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def resize_square_image(
    img: Image, width: int = 640, background_color: tuple = (0, 0, 0)
):
    if img.mode != "RGB":
        return None
    img = expand2square(img, (0, 0, 0))
    img = img.resize((width, width))

    return img


def get_original_xy(
    xy: tuple, image_size_original: tuple, image_size_new: tuple
) -> tuple:
    """Every point is from the PIL Image point of view.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    x, y = xy[0], xy[1]

    width_original = image_size_original[0]
    height_original = image_size_original[1]

    width_new = image_size_new[0]
    height_new = image_size_new[1]

    if width_original > height_original:

        x = width_original / width_new * x
        y = width_original / width_new * y

        bias = (width_original - height_original) / 2
        y = y - bias

    else:

        x = height_original / height_new * x
        y = height_original / height_new * y

        bias = (height_original - width_original) / 2
        x = x - bias

    return x, y


def get_original_bbox(
    bbox: np.ndarray, image_size_original: tuple, image_size_new: tuple
) -> np.ndarray:
    """Get the original coordinates of the bounding box.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    bbox_new = []
    for xy in [bbox[:2], bbox[2:]]:
        xy = xy[0], xy[1]
        xy = get_original_xy(xy, image_size_original, image_size_new)
        bbox_new.append(xy)

    bbox_new = [bar for foo in bbox_new for bar in foo]
    bbox_new = np.array(bbox_new)

    return bbox_new


def get_original_lm(
    lm: np.ndarray, image_size_original: tuple, image_size_new: tuple
) -> np.ndarray:
    """Get the original coordinates of the five landmarks.

    Note that image_size_original and image_size_new are PIL Image size values.
    """
    lm_new = []
    for lm_ in lm:
        xy = lm_[0], lm_[1]
        xy = get_original_xy(xy, image_size_original, image_size_new)
        lm_new.append(xy)

    lm_new = np.array(lm_new)

    return lm_new


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def load_Adience_labels():
    logging.info(f"loading adience labels ...")
    folds = []
    for i in tqdm(range(5)):
        with open(f"./data/Adience/fold_{i}_data.txt", "r") as stream:
            fold = stream.readlines()
        fold = [line.strip().split("\t") for line in fold]
        header = fold[0]
        fold = fold[1:]
        folds.append(fold)

    return folds, header


def get_Adience_image_paths(image_type: str = "aligned", resize: bool = False):
    logging.info(f"getting Adience image paths ...")

    folds, header = load_Adience_labels()

    ages = []
    genders = []
    image_paths = []
    fold_from = []
    logs = {}
    for idx, fold in tqdm(enumerate(folds)):
        for row in fold:
            user_id = row[0]
            original_image = row[1]
            face_id = row[2]
            age = row[3]
            gender = row[4]

            if image_type == "raw":
                image_path = os.path.join(
                    f"data/Adience/faces/{user_id}/coarse_tilt_aligned_face.{face_id}.{original_image}"
                )
            elif image_type == "aligned":
                image_path = os.path.join(
                    f"data/Adience/aligned/{user_id}/landmark_aligned_face.{face_id}.{original_image}"
                )
            else:
                raise ValueError

            if resize:
                assert os.path.isfile(image_path + ".RESIZED.jpg")
            else:
                assert os.path.isfile(image_path)

            ages.append(age)
            genders.append(gender)
            image_paths.append(image_path)
            fold_from.append(idx)

    assert len(image_paths) == len(ages) == len(genders)

    logging.info(f"In total of {len(image_paths)} found!")
    logs["total_num_images"] = len(image_paths)

    return image_paths, folds, header, ages, genders, fold_from, logs


def choose_one_face(image_path: str, list_of_fdr: list, method: str = "center"):
    logging.debug(f"number of faces is {len(list_of_fdr)}")

    if method == "biggest":
        bbox_size = [fdr["bbox"] for fdr in list_of_fdr]
        bbox_size = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bbox_size]
        idx = np.argmax(bbox_size)
        fdr = list_of_fdr[idx]

    elif method == "center":
        img = Image.open(image_path)
        width, height = img.size
        image_center = height // 2, width // 2
        bbox_centers = [fdr["bbox"] for fdr in list_of_fdr]
        bbox_centers = [
            ((bbox[2] + bbox[0]) // 2, (bbox[3] - bbox[1]) // 2)
            for bbox in bbox_centers
        ]
        logging.debug(f"{bbox_centers}, {image_center}")
        dists = [
            np.linalg.norm(np.array(bbox) - np.array(image_center))
            for bbox in bbox_centers
        ]
        idx = np.argmin(dists)
        fdr = list_of_fdr[idx]
    else:
        raise ValueError

    return fdr


def get_nearest_number(
    query: float, predefined: list = [28.5, 40.5, 5.0, 80.0, 17.5, 50.5, 10.0, 1.0]
):
    # possible ages
    # 28.5
    # 40.5
    # 5.0
    # 80.0
    # 17.5
    # 50.5
    # 10.0
    # 1.0
    if query is None:
        return query
    diffs = [np.abs(query - pre) for pre in predefined]
    idx = np.argmin(diffs)

    return predefined[idx]


def remove_nones_Adience(
    image_paths: list,
    ages: list,
    genders: list,
    fold_from: list,
    logs: list = None,
    det_score: float = 0.90,
):

    logging.info(f"removing Nones from the data ...")

    assert len(image_paths) == len(ages) == len(genders) == len(fold_from)
    image_paths_, ages_, genders_, fdr_paths_, fold_from_, embeddings_ = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    removals = {}
    removals["no_image_path"] = 0
    removals["no_age"] = 0
    removals["no_gender"] = 0
    removals["no_fold"] = 0
    removals["no_embeddings"] = 0
    removals["no_face_detected"] = 0
    removals["bad_quality"] = 0

    for i, j, k, l in tqdm(zip(image_paths, ages, genders, fold_from)):
        if i is None:
            removals["no_image_path"] += 1
            continue
        if j is None:
            removals["no_age"] += 1
            continue
        if k is None:
            removals["no_gender"] += 1
            continue
        if l is None:
            removals["no_fold"] += 1
            continue
        fdr_path = i + ".pkl"

        if not os.path.isfile(fdr_path):
            removals["no_embeddings"] += 1
            continue

        with open(fdr_path, "rb") as stream:
            embedding = pickle.load(stream)

        if len(embedding) == 0:
            removals["no_face_detected"] += 1
            continue

        elif len(embedding) == 1:
            embedding = embedding[0]
        else:
            embedding = choose_one_face(i, embedding, "center")

        if embedding["det_score"] < det_score:
            removals["bad_quality"] += 1
            continue

        image_paths_.append(i)
        ages_.append(j)
        genders_.append(k)
        fdr_paths_.append(fdr_path)
        fold_from_.append(l)
        embeddings_.append(embedding)

    assert (
        len(image_paths_)
        == len(ages_)
        == len(genders_)
        == len(fdr_paths_)
        == len(fold_from_)
        == len(embeddings_)
    )

    logging.warning(f"some data removed: {removals}")

    if logs is not None:
        logs["removed"] = removals

    return image_paths_, ages_, genders_, fdr_paths_, fold_from_, embeddings_, logs


def calc_age(taken, dob):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    assert (
        len(full_path)
        == len(dob)
        == len(gender)
        == len(photo_taken)
        == len(face_score)
        == len(second_face_score)
        == len(age)
    )
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    """Copied from
    https://github.com/yu4u/age-gender-estimation/blob/master/src/utils.py
    """
    d = loadmat(mat_path)

    return (
        d["image"],
        d["gender"][0],
        d["age"][0],
        d["db"][0],
        d["img_size"][0, 0],
        d["min_score"][0, 0],
    )
