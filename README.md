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

The port number `10002` is the port that the insightface docker container listens to. Set `cuda=True` in the below code snippet, if you want to run on a NVIDIA GPU). The face embedding vectors are pickled. They are saved as `image-path.pkl` (e.g. `landmark_aligned_face.2174.9523333835_c7887c3fde_o.jpg.RESIZED.pkl`). 

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

The model is an MLP with residual connections. It's very light.

### Training steps

There are three training steps involved.

1. Hyperparameter search using [Ray Tune](https://docs.ray.io/en/master/tune/index.html)

    This searches dropout rate, number of residuals per block, number of blocks in the network, batch size, peak learning rate, weight decay rate, and gamma of exponential learning rate decay. See `hp-tuning.py` and `hp-tuning.json` for the details.

1. Pre-training on the `IMDB` and `WIKI` dataset. 

    We'll use the optimal hyperparameters found in the step 1 to pre-train the model. See `train.py` and `train.json` for the details.

1. Five random seeds on 5-fold cross-validation on the `Adience` dataset. 

    Since the reported metrics (i.e. accuracy) is 5-fold cross-validation, we will do the same here. In order to get the least biased numbers, we run this five times each with a different seed. This means that we are training in total of 25 times and report the average of the 25 numbers. See `cross-val.json` for the details.

## Evaluation results

Validation split is always 10%

### Gender (2 classes, cross entropy loss)

#### Train on IMDB and WIKI

```
train_loss:     0.4422892515230485
train_accuracy: 0.8247924730961401

val_loss:       0.44423754876240706
val_accuracy:   0.8241741959549072
```

#### Train on IMDB, WIKI, and Adience

```
train_loss:     0.42638325623907836
train_accuracy: 0.831892218514716

val_loss:       0.4287793638150385
val_accuracy:   0.8336972114977788
```

#### Pre-trained on IMDB and WIKI, fine-tuned on Adience (5 times 5-fold cross validation)

```
"train_loss_mean":      0.03909929594595114,
"train_loss_std":       0.002372652796471939,
"train_accuracy_mean":  0.9889659560623849,
"train_accuracy_std":   0.0008610543612170836,

"val_loss_mean":        0.08575112012914929,
"val_loss_std":         0.016764874765751312,
"val_accuracy_mean":    0.9754258021389498,
"val_accuracy_std":     0.004659287766331918,

"test_loss_mean":       0.26193851897021414,
"test_loss_std":        0.052632717117045105,
"test_accuracy_mean":   0.908585836031218,
"test_accuracy_std":    0.02015994359625317
```

#### Train on Adience from scratch (5 times 5-fold cross validation)

```
"train_loss_mean":      0.04796776738837142,
"train_loss_std":       0.003088481406336201,
"train_accuracy_mean":  0.9865555643047109,
"train_accuracy_std":   0.0011906210955395275,

"val_loss_mean":        0.10044357062645601,
"val_loss_std":         0.01445883209039448,
"val_accuracy_mean":    0.9690235843266057,
"val_accuracy_std":     0.004473660949098477,

"test_loss_mean":       0.3757506806973789,
"test_loss_std":        0.07711257385288053,
"test_accuracy_mean":   0.8545809620850371,
"test_accuracy_std":    0.030167924692375468
```

### Age (8 classes, cross entropy loss)

#### Train on IMDB and WIKI

```
train_loss:     1.0887723597771908
train_accuracy: 0.5783940968645709

val_loss:       1.1134433766429344
val_accuracy:   0.570466669813993
```

#### Train on IMDB, WIKI, and Adience

```
train_loss:     1.0673519349163514
train_accuracy: 0.5878167265879131

val_loss:       1.0959079291309892
val_accuracy:   0.5741907310982413
```

#### Pre-trained on IMDB and WIKI, fine-tuned on Adience (5 times 5-fold cross validation)

```
"train_loss_mean":      0.20047779845682956,
"train_loss_std":       0.011232020610773797,
"train_accuracy_mean":  0.9425886618487421,
"train_accuracy_std":   0.0036201164737907085,

"val_loss_mean":        0.31623824133047923,
"val_loss_std":         0.011597632628809347,
"val_accuracy_mean":    0.891788721381898,
"val_accuracy_std":     0.0049177913990907695,

"test_loss_mean":       1.0386354841263716,
"test_loss_std":        0.06812033572619748,
"test_accuracy_mean":   0.6122717854711083,
"test_accuracy_std":    0.03492616380546068
```

#### Train on Adience from scratch (5 times 5-fold cross validation)

```
"train_loss_mean":      0.17211890350209344,
"train_loss_std":       0.009164870065452248,
"train_accuracy_mean":  0.9531629083524806,
"train_accuracy_std":   0.003472722375862844,

"val_loss_mean":        0.3142906121185102,
"val_loss_std":         0.025120475593731324,
"val_accuracy_mean":    0.8997177752525727,
"val_accuracy_std":     0.009694706998382105,

"test_loss_mean":       1.2816829867808728,
"test_loss_std":        0.11533204273999496,
"test_accuracy_mean":   0.5499599883287748,
"test_accuracy_std":    0.039982039677149694
```        

### Age (101 classes, cross entropy loss)

#### Train on IMDB and WIKI

```
train_loss:             3.3405551750547704
train_accuracy:         0.1458298327252277
train_accuracy_relaxed: 0.5593033732716998

val_loss:               3.4325964542535634
val_accuracy:           0.12708819148043352
val_accuracy_relaxed:   0.5443172292625807
```
#### Train on IMDB, WIKI, and Adience

```
train_loss:             3.2645355109150858
train_accuracy:         0.16898405437756497
train_accuracy_relaxed: 0.5740050017099864

val_loss:               3.350942190100507
val_accuracy:           0.1496662111017662
val_accuracy_relaxed:   0.5579260079373424
```

#### Pre-trained on IMDB and WIKI, fine-tuned on Adience (5 times 5-fold cross validation)

```
"train_loss_mean":              0.8063892030820919,
"train_loss_std":               0.03660549208896196,
"train_accuracy_mean":          0.8096710611161348,
"train_accuracy_std":           0.0145801494205596,
"train_accuracy_relaxed_mean":  0.8099365367156465,
"train_accuracy_relaxed_std":   0.014508487444787933,

"val_loss_mean":                0.9433291613735717,
"val_loss_std":                 0.03815652115669657,
"val_accuracy_mean":            0.766776820749713,
"val_accuracy_std":             0.009086219922726001,
"val_accuracy_relaxed_mean":    0.7669245312371575,
"val_accuracy_relaxed_std":     0.009035461777985048,

"test_loss_mean":               1.7269022825381142,
"test_loss_std":                0.2489200009428155,
"test_accuracy_mean":           0.5357882123484939,
"test_accuracy_std":            0.038684415614598566,
"test_accuracy_relaxed_mean":   0.5361665770147007,
"test_accuracy_relaxed_std":    0.03888578643764644
```

#### Train on Adience from scratch (5 times 5-fold cross validation)

```
"train_loss_mean":              0.9313781216021964,
"train_loss_std":               0.050226683769426315,
"train_accuracy_mean":          0.7560901150002954,
"train_accuracy_std":           0.011359064308840948,
"train_accuracy_relaxed_mean":  0.7560901150002954,
"train_accuracy_relaxed_std":   0.011359064308840948,

"val_loss_mean":                1.005587027235029,
"val_loss_std":                 0.04576162941291698,
"val_accuracy_mean":            0.739558046004143,
"val_accuracy_std":             0.01351175844198754,
"val_accuracy_relaxed_mean":    0.739558046004143,
"val_accuracy_relaxed_std":     0.01351175844198754,

"test_loss_mean":               1.7334976813679328,
"test_loss_std":                0.1963164904447087,
"test_accuracy_mean":           0.5186810812340403,
"test_accuracy_std":            0.045878284812807146,
"test_accuracy_relaxed_mean":   0.5186810812340403,
"test_accuracy_relaxed_std":    0.045878284812807146
```

## Deployment

We also provide a production-ready model and code. This model is trained on all of the three datasets (i.e. `Adience`, `IMDB`, and `WIKI`). The training was done on the random 90% of the samples and the remaining 10% of the samples were used as a validation split. There is no test split, since we aren't reporting to some benchmark.

The deployed app is a simple flask server app with one endpoint. This endpoint takes one 512-dimensional arcface embedding vector and outputs the gender probability and estimated age.  

We also wrap this with a docker container. There are two docker files. `Dockerfile` is for CPU-only and `Dockerfile-cuda` is for GPU. The models are very lightweight. You probably don't need a GPU for this.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Authors

* [Taewoon Kim](https://taewoonkim.com/) 