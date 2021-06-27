# age-gender

This repo contains code to train age / gender prediction and run inference on a flask server. The pytorch model training / testing was copied using [this template](https://github.com/victoresque/pytorch-template).

## Datasets

I used [Adience age and gender dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html). Download the data and place them at `./data/Adience/`.

You can find the state of the art models at [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age) for the age and [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience) for the gender, respectively.

## Training

I advise you that you run all of below in a virutal python environment.

My first approach is to use arcface face embedding vectors. I think they might include some features that are relevant for age and gender.

### Build the insightface docker container.

The insturctions can be found [here](https://github.com/Taeert/insightface/blob/main/README-taeert.md). 

### Extract the face embeddings.

At the root of this repo, run the below command

```bash
python3 -c "from utils.scripts import extract_Adience_arcface; extract_Adience_arcface('aligned', 10002)"
```

It might take some time. The argument `aligned` means that we'll be using the aligned face images, not raw. The aligned images have the face of interest in the center, which makes it easier to find the face. The port number `10002` is the port that the insightface docker container listens to.


### Save the data and metadata.

```bash
python3 -c "from utils.scripts import get_Adience_clean; get_Adience_clean('aligned')"
```

This will write `./data/Adience/meta-data-aligned.json` and `./data/Adience/data-aligned.npy`

### Training a model

I'll try MLP first.

### Evaluation results

The Adience has five folds of data. The reported metrics are the mean values of cross-validation accuracy from the five test splits.

#### Adience-gender

| leave-one-out | train              | val                | test               |
| ------------- | ------------------ | ------------------ | ------------------ |
| 0             | 0.9227379643206257 | 0.9429201555023923 | 0.7994943109987358 |
| 1             | 0.9082845052083334 | 0.9459087401795735 | 0.9542351453855878 |
| 2             | 0.8894158291457287 | 0.9332386363636364 | 0.9443742098609356 |
| 3             | 0.9030612244897959 | 0.9500210437710437 | 0.9491782553729456 |
| 4             | 0.8984999605512307 | 0.9365920805998126 | 0.9504424778761061 |
| mean          | 0.904399896743143  | 0.941736131283292  | 0.919544879898862  |
		

#### Adience-age
| leave-one-out | train              | val                | test               |
| ------------- | ------------------ | ------------------ | ------------------ |
| 0             | 0.5629353005865102 | 0.6214862440191388 | 0.4718078381795196 |
| 1             | 0.6153738839285714 | 0.6878945707070707 | 0.770417193426043  |
| 2             | 0.6282506281407035 | 0.6605113636363636 | 0.6950695322376739 |
| 3             | 0.6009247448979592 | 0.6254997895622896 | 0.6986093552465233 |
| 4             | 0.6320777666736798 | 0.6591568064667291 | 0.7494310998735777 |
| mean          | 0.607912464845485  | 0.650909754878318  | 0.677067003792667  |
		

## Authors

* [Taewoon Kim](https://taewoonkim.com/) 