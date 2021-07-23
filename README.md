# age-gender

This repo contains code to train age / gender prediction and run inference on a flask server. The pytorch model training / testing was copied using [this template](https://github.com/victoresque/pytorch-template).

## Datasets

I used [Adience age and gender dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html). Download the data and place them at `./data/Adience/`.

You can find the state of the art models at [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age) for the age and [here](https://paperswithcode.com/sota/age-and-gender-classification-on-adience) for the gender, respectively.

I also used [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). This dataset is huge. It's got more than 500,000 faces with gender and age labeled. One weird thing is that this dataset doesn't have train / val / test splits. This dataset is pretty much only used to pre-train your model. I don't know why but that is what it is. People don't compare their scores against this dataset. So I'll do the same. I'll use this dataset to improve my training and report the final metrics on the Adience age and gender dataset. Anyways, download `imdb_crop.tar` and `wiki_crop.tar`, and place them at `./data/imdb_crop` and `./data/wiki_crop`, respectively. 

## Data pre-processing

I advise you that you run all of below in a virutal python environment.

My first approach is to use arcface face embedding vectors. I think they might include some features that are relevant for age and gender.

### Build the face-detection-recognition docker container.

The insturctions can be found [here](https://github.com/tae898/face-detection-recognition/blob/main/README.md). 

### Extract the arcface face embeddings.

It might take some time. 

The port number `10002` is the port that the face-detection-recognition docker container listens to. Set `cuda=True` in the below code snippet, if you want to run on a NVIDIA GPU). The face embedding vectors are pickled. They are saved as `<image-path>.pkl` (e.g. `landmark_aligned_face.2174.9523333835_c7887c3fde_o.jpg.RESIZED.pkl`). 

Resizing image to the same shape (e.g. `resize=640` resizes every image to a black background square RGB image with the width and height being 640 pixels) dramatically increase the speed due to some mxnet stuff that I'm not a big fan of.

`det_score` is the confidence score on face detection. The faces whose confidence score is lower than this this threshold value will not be considered.

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
    python3 -c "from utils.scripts import get_imdb_wiki_clean; get_imdb_wiki_clean('imdb', resize=640, det_score=0.9)"
    ```

    This will write `./data/imdb_crop/meta-data.json` and `./data/imdb_crop/data.npy`

3. WIKI

    At the root of this repo, run the below command.

    ```bash
    python3 -c "from utils.scripts import extract_imdb_wiki_arcface; extract_imdb_wiki_arcface('wiki', docker_port=10002, cuda=False, resize=640)"
    ```

    ```bash
    python3 -c "from utils.scripts import get_imdb_wiki_clean; get_imdb_wiki_clean('wiki', resize=640, det_score=0.9)"
    ```

    This will write `./data/wiki_crop/meta-data.json` and `./data/wiki_crop/data.npy`

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

    Genders
    | female | male  |
    | ------ | ----- |
    | 9,103  | 7,952 |

    Ages
    | 0 to 2 | 4 to 6 | 8 to 12 | 15 to 20 | 25 to 32 | 38 to 43 | 48 to 53 | 60 to 100 |
    | ------ | ------ | ------- | -------- | -------- | -------- | -------- | --------- |
    | 1,363  | 2,087  | 2,226   | 1,761    | 5,162    | 2,719    | 907      | 830       |


2. IMDB age and gender dataset

    This dataset does not have train / val / test splits. Researchers normally use this dataset for pretraining.

    | images before removal |
    | --------------------- |
    | 460,723               |

    Removed data
    | failed to process image | no age found | no gender found | no face detected | more than one face | bad quality (det_score<0.9) | no embeddings | SUM               |
    | ----------------------- | ------------ | --------------- | ---------------- | ------------------ | --------------------------- | ------------- | ----------------- |
    | 22,200                  | 690          | 8,453           | 21,441           | 47,278             | 3855                        | 27            | 103,944 (22.56 %) |

    Genders
    | female  | male    |
    | ------- | ------- |
    | 153,316 | 203,463 |

    Ages

    Ages are fine-grained integers from 0 to 100. Check `./data/imdb_crop/meta-data.json` for the details.

3. WIKI age and gender dataset

    This dataset does not have train / val / test splits. Researchers normally use this dataset for pretraining.

    | images before removal |
    | --------------------- |
    | 62,328                |

    Removed data
    | failed to process image | no age found | no gender found | no face detected | more than one face | bad quality (det_score<0.9) | no embeddings | SUM              |
    | ----------------------- | ------------ | --------------- | ---------------- | ------------------ | --------------------------- | ------------- | ---------------- |
    | 10,909                  | 1,781        | 2,485           | 3,074            | 2,179              | 428                         | 0             | 20,856 (33.46 %) |

    Genders
    | female | male   |
    | ------ | ------ |
    | 9,912  | 31,560 |

    Ages

    Ages are fine-grained integers from 0 to 100. Check `./data/imdb_crop/meta-data.json` for the details.

## Training

### Model

The model is basically an MLP. There are two variants considered. One is a pure MLP and the other is MLP with [IC layers](https://arxiv.org/pdf/1905.05928.pdf). It's emperically shown that the latter is better than the pure MLP.

### Training steps

There are three training steps involved.

1. Hyperparameter search using [Ray Tune](https://docs.ray.io/en/master/tune/index.html)

    This searches dropout rate, number of residuals per block, number of blocks in the network, batch size, peak learning rate, weight decay rate, and gamma of exponential learning rate decay. Configure the values in `hp-tuning.json` and run `python hp-tuning.py`.

2. Pre-training on the `IMDB` and `WIKI` dataset. 

    We'll use the optimal hyperparameters found in the step 1 to pre-train the model. Configure the values in `train.json` and run `python train.py`.

3. Five random seeds on 5-fold cross-validation on the `Adience` dataset. 

    Since the reported metrics (i.e. accuracy) is 5-fold cross-validation, we will do the same here. In order to get the least biased numbers, we run this five times each with a different seed. This means that we are training in total of 25 times and report the average of the 25 numbers. Configure the values in `cross-val.json` and run `python cross-val.py`.

## [Evaluation results](training-results/TRAINING-RESULTS.md)

Click on the above link to see the detailed results.

## Qualitative analysis

Check `./test-images` to see the model inference results on some stock images.

## Running and deploying the models

We provide the gender and the age models, which are trained on IMDB, WIKI, and Adience datasets. The gender model is a binary classification and the age model is a 101-class (from 0 to 100 years old) classification. They are MLPs with dropout, batch norm, and residual connections. They can be found at `./models/gender.pth` and `./models/age.pth`, respectively. Both are light-weight. Running on a CPU is enough.

`app.py` is a flask server app that receives accepts 512-dimensional arcface embeddings and returns estimated genders and ages. You can also run this on a docker container. Build the container by running

```
docker build -t age-gender .
```

After running the container (i.e. `docker run -it --rm -p 10003:10003 age-gender`), you can run `client.py` (e.g. `python client.py --image-path test-images/matrix-tae-final_exported_37233.jpg`) to get estimated genders and ages in the picture. 

Note that you also have to run the face-detection-recognition (`docker run -it --rm -p 10002:10002 face-detection-recognition` for CPU or `docker run --gpus all -it --rm -p 10002:10002 face-detection-recognition-cuda` for cuda), before running `client.py`. This separation might be annoying but the modularization will help in the future.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Authors

* [Taewoon Kim](https://taewoonkim.com/) 