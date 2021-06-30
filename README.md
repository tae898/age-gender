# age-gender

This repo contains code to train age / gender prediction and run inference on a flask server. The pytorch model training / testing was copied using [this template](https://github.com/victoresque/pytorch-template).

## Datasets

I used [Adience age and gender dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html). Download the data and place them at `./data/Adience/`.

You can find the state of the art models at [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age) for the age and [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience) for the gender, respectively.

I also used [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). This dataset is huge. It's got more than 500,000 faces with gender and age labeled. One weird thing is that this dataset doesn't have train / val / test splits. This dataset is pretty much only used to pre-train your model. I don't know why but that is what it is. People don't compare their scores against this dataset. So I'll do the same. I'll use this dataset to improve my training and report the final metrics on the Adience age and gender dataset. Anyways, download `imdb_crop.tar` and `wiki_crop.tar`, and place them at `./data/imdb_crop` and `./data/wiki_crop`, respectively. 

## Data pre-processing

I advise you that you run all of below in a virutal python environment.

My first approach is to use arcface face embedding vectors. I think they might include some features that are relevant for age and gender.

### Build the insightface docker container.

The insturctions can be found [here](https://github.com/taeert/insightface/blob/main/README-taeert.md). 

### Extract the arcface face embeddings.

It might take some time. 

The port number `10002` is the port that the insightface docker container listens to. Set `cuda=True` if you want to run on a NVIDIA GPU). The face embedding vectors are pickled. They are saved as `image-path.pkl` (e.g. `landmark_aligned_face.2174.9523333835_c7887c3fde_o.jpg.RESIZED.pkl`). 

Resizing image to the same shape (e.g. `resize=640` resizes every image to a black background square RGB image with the width and height being 640 pixels) dramatically increase the speed due to some mxnet stuff that I'm not a big fan of.

`det_score` is the confidence score on face detection. The faces whose confidence score is lower than this this value will not be considered.

1.  Adience age and gender dataset

    At the root of this repo, run the below command.

    ```bash
    python3 -c "from utils.scripts import extract_Adience_arcface; extract_Adience_arcface('aligned', docker_port=10002, cuda=False, resize=640)"
    ```
    The argument `aligned` means that we'll be using the aligned face images, not raw. The aligned images have the face of interest in the center, which makes it easier to find the face. 

    ```bash
    python3 -c "from utils.scripts import get_Adience_clean; get_Adience_clean('aligned', resize=640, det_score=0.9)"
    ```

    This will write `./data/Adience/meta-data-aligned.json` and `./data/Adience/data-aligned.npy`

2. IMDB

    At the root of this repo, run the below command.

    ```bash
    python3 -c "from utils.scripts import extract_imdb_wiki_arcface; extract_imdb_wiki_arcface('imdb', docker_port=10002, cuda=False, resize=640)"
    ```

    ```bash
    python3 -c "from utils.scripts import get_imdb_wiki_clean; get_imdb_wiki_clean('imdb', resize=640)"
    ```

    This will write `./data/imdb_crop/imdb.csv`

3. WIKI

    At the root of this repo, run the below command.

    ```bash
    python3 -c "from utils.scripts import extract_imdb_wiki_arcface; extract_imdb_wiki_arcface('wiki', docker_port=10002, cuda=False, resize=640)"
    ```

    ```bash
    python3 -c "from utils.scripts import get_imdb_wiki_clean; get_imdb_wiki_clean('wiki', resize=640)"
    ```

    This will write `./data/wiki_crop/wiki.csv`

### Dataset stats

1. Adience age and gender dataset

    This dataset has five folds. The performance metric is accuracy on five-fold cross validation.

    | images before removal | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
    | --------------------- | ------ | ------ | ------ | ------ | ------ |
    | 19,370                | 4,484  | 3,730  | 3,894  | 3,446  | 3,816  |

    Removed data
    | failed to process image | no age found | no gender found | no face detected | bad quality (det_score<0.9) | SUM             |
    | ----------------------- | ------------ | --------------- | ---------------- | --------------------------- | --------------- |
    | 0                       | 748          | 1,170           | 322              | 75                          | 2,315 (11.95 %) |

    Genders and ages
    | female | male  | 0 to 2 | 4 to 6 | 8 to 12 | 15 to 20 | 25 to 32 | 38 to 43 | 48 to 53 | 60 to 100 |
    | ------ | ----- | ------ | ------ | ------- | -------- | -------- | -------- | -------- | --------- |
    | 9,103  | 7,952 | 1,363  | 2,087  | 2,226   | 1,761    | 5,162    | 2,719    | 907      | 830       |


### Training

#### Training a model

1. MLP
    
    wow this works good.

1. MLP + Residual

    Let's see

1. MLP + Residual + automatic HP tuning

    Let's see


### Evaluation results

The Adience has five folds of data. The reported metrics are the mean values of cross-validation accuracy from the five test splits.

#### Adience-gender (no additional training data)

| leave-one-out | train              | val                | test               |
| ------------- | ------------------ | ------------------ | ------------------ |
| 0             | 0.9227379643206257 | 0.9429201555023923 | 0.7994943109987358 |
| 1             | 0.9082845052083334 | 0.9459087401795735 | 0.9542351453855878 |
| 2             | 0.8894158291457287 | 0.9332386363636364 | 0.9443742098609356 |
| 3             | 0.9030612244897959 | 0.9500210437710437 | 0.9491782553729456 |
| 4             | 0.8984999605512307 | 0.9365920805998126 | 0.9504424778761061 |
| mean          | 0.904399896743143  | 0.941736131283292  | 0.919544879898862  |
| std           | 0.012374353292582  | 0.006821309086719  | 0.067202767320067  |

#### Adience-age (no additional training data)

| leave-one-out | train              | val                | test               |
| ------------- | ------------------ | ------------------ | ------------------ |
| 0             | 0.5629353005865102 | 0.6214862440191388 | 0.4718078381795196 |
| 1             | 0.6153738839285714 | 0.6878945707070707 | 0.770417193426043  |
| 2             | 0.6282506281407035 | 0.6605113636363636 | 0.6950695322376739 |
| 3             | 0.6009247448979592 | 0.6254997895622896 | 0.6986093552465233 |
| 4             | 0.6320777666736798 | 0.6591568064667291 | 0.7494310998735777 |
| mean          | 0.607912464845485  | 0.650909754878318  | 0.677067003792667  |
| std           | 0.027951068889436  | 0.027565788040492  | 0.119237482897551  |
		

## Authors

* [Taewoon Kim](https://taewoonkim.com/) 