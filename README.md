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

## Authors

* [Taewoon Kim](https://taewoonkim.com/) 