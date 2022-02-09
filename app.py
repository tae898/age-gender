import io
import logging

import jsonpickle
import numpy as np
import torch
from flask import Flask, request
from tqdm import tqdm

from model.model import ResMLP
from utils import enable_dropout, forward_mc, read_json

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# a light-weight flask app
app = Flask(__name__)

models = {"age": None, "gender": None}

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

for model_ in ["age", "gender"]:
    model = ResMLP(**read_json(f"./models/{model_}.json")["arch"]["args"])
    checkpoint = f"models/{model_}.pth"
    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    enable_dropout(model)

    models[model_] = model

# One endpoint is enough.


@app.route("/", methods=["POST"])
def predict_age_gender():
    """Receive everything in json!!!"""
    app.logger.debug(f"Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    app.logger.debug(f"loading embeddings ...")
    embeddings = data["embeddings"]
    embeddings = io.BytesIO(embeddings)

    # This assumes that the client has serialized the embeddings with pickle.
    # before sending it to the server.
    embeddings = np.load(embeddings, allow_pickle=True)

    # -1 accounts for the batch size.
    embeddings = embeddings.reshape(-1, 512).astype(np.float32)

    app.logger.debug(f"extracting gender and age from {embeddings.shape[0]} faces ...")

    genders = []
    ages = []

    for embedding in tqdm(embeddings):
        embedding = embedding.reshape(1, 512)
        gender_mean, gender_entropy = forward_mc(models["gender"], embedding)
        age_mean, age_entropy = forward_mc(models["age"], embedding)
        gender = {"m": 1 - gender_mean, "f": gender_mean, "entropy": gender_entropy}
        age = {"mean": age_mean, "entropy": age_entropy}

        genders.append(gender)
        ages.append(age)

    app.logger.debug(f"gender and age extracted!")

    response = {"ages": ages, "genders": genders}

    response_pickled = jsonpickle.encode(response)
    app.logger.info("json-pickle is done.")

    return response_pickled


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=10003)
