from flask import Flask, request
import jsonpickle
import logging
import numpy as np
import io
from model.model import ResMLP
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

app = Flask(__name__)

device = torch.device('cpu')
model_gender = ResMLP(dropout=0.5, num_residuals_per_block=3, num_blocks=1,
                      num_classes=2, num_initial_features=512, last_activation=None,
                      min_bound=None, max_bound=None, only_MLP=False)

checkpoint = "models/gender.pth"
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
model_gender.load_state_dict(state_dict)
model_gender.to(device)
model_gender.eval()

model_age = ResMLP(dropout=0.2, num_residuals_per_block=4, num_blocks=0,
                   num_classes=101, num_initial_features=512, last_activation=None,
                   min_bound=None, max_bound=None, only_MLP=False)

checkpoint = "models/age.pth"
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
model_age.load_state_dict(state_dict)
model_age.to(device)
model_age.eval()


@app.route("/", methods=["POST"])
def extract_frames():
    """
    Receive everything in json!!!

    """
    app.logger.debug(f"Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)

    app.logger.debug(f"loading embeddings ...")
    embeddings = data['embeddings']
    embeddings = io.BytesIO(embeddings)

    embeddings = np.load(embeddings, allow_pickle=True)

    embeddings = embeddings.reshape(-1, 512).astype(np.float32)
    embeddings = torch.tensor(embeddings)

    app.logger.debug(f"extracting gender and age ...")
    genders = model_gender(embeddings)
    ages = model_age(embeddings)

    genders = torch.softmax(genders, dim=1)
    genders = genders.detach().cpu().numpy()
    genders = [{'m': gender[0].item(), 'f': gender[1].item()}
               for gender in genders]

    ages = torch.softmax(ages, dim=1)
    ages = torch.argmax(ages, dim=1)
    ages = ages.detach().cpu().numpy()
    ages = [np.arange(0, 101)[age].item() for age in ages]

    app.logger.debug(f"{len(ages)} gender and age extracted!")

    response = {'ages': ages, 'genders': genders}

    response_pickled = jsonpickle.encode(response)
    app.logger.info("json-pickle is done.")

    return response_pickled


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=10003)
