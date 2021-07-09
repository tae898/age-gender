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

    Since the reported metrics (i.e. accuracy) is 5-fold cross-validation, we will do the same here. In order to get the least biased numbers, we run this five times each with a different seed. This means that we are training in total of 25 times and report the average of the 25 numbers. 

## Evaluation results

### Gender

#### Pre-trained
```
"train_loss_mean": 0.035756768169454604,
"train_loss_std": 0.0020211418618862223,
"train_accuracy_mean": 0.9902959872287564,
"train_accuracy_std": 0.0007618696495071593,

"val_loss_mean": 0.03806625073733362,
"val_loss_std": 0.00932198958435307,
"val_accuracy_mean": 0.9899641804867985,
"val_accuracy_std": 0.0028286318037089556,

"test_loss_mean": 0.25444767223690967,
"test_loss_std": 0.06082393270949546,
"test_accuracy_mean": 0.9109055097220713,
"test_accuracy_std": 0.022389514578131632
```

#### Not pre-trained

```
"train_loss_mean": 0.04613704621584779,
"train_loss_std": 0.003341978420252909,
"train_accuracy_mean": 0.9862985664676904,
"train_accuracy_std": 0.001101129657443983,

"val_loss_mean": 0.04566914670182671,
"val_loss_std": 0.007143097508032387,
"val_accuracy_mean": 0.9867329020619238,
"val_accuracy_std": 0.002521348547538151,

"test_loss_mean": 0.39788186767442213,
"test_loss_std": 0.08673959940676833,
"test_accuracy_mean": 0.855026333055,
"test_accuracy_std": 0.030152854986659282
```

### Age (8-class)

#### Pre-trained

```
"train_loss_mean": 0.266580416006805,
"train_loss_std": 0.009325988423167151,
"train_accuracy_mean": 0.9226152428724472,
"train_accuracy_std": 0.0031907232333313407,

"val_loss_mean": 0.3680720592439847,
"val_loss_std": 0.017673356303800316,
"val_accuracy_mean": 0.874282163736166,
"val_accuracy_std": 0.008070140340575006,

"test_loss_mean": 1.0014392203764806,
"test_loss_std": 0.07049642859712374,
"test_accuracy_mean": 0.615897137353467,
"test_accuracy_std": 0.035343900485183065
```

#### Not pre-trained

```
"train_loss_mean": 0.2676352219275762,
"train_loss_std": 0.014189184219480759,
"train_accuracy_mean": 0.9261217717210379,
"train_accuracy_std": 0.004780756026425679,

"val_loss_mean": 0.3983086254958766,
"val_loss_std": 0.02044003718494854,
"val_accuracy_mean": 0.8712536503832434,
"val_accuracy_std": 0.008218882242723824,

"test_loss_mean": 1.2667344478898865,
"test_loss_std": 0.1342585171007094,
"test_accuracy_mean": 0.541302246287051,
"test_accuracy_std": 0.03968895966501664
```

### Age (101-class)

#### Pre-trained

```
"train_loss_mean": 0.2177276595641424,
"train_loss_std": 0.00937389201247591,
"train_accuracy_mean": 0.9369813516215164,
"train_accuracy_std": 0.003651313385947366,
"train_accuracy_relaxed_mean": 0.9369813516215164,
"train_accuracy_relaxed_std": 0.003651313385947366,

"val_loss_mean": 0.33898204408896293,
"val_loss_std": 0.02523015861325991,
"val_accuracy_mean": 0.8866003916678595,
"val_accuracy_std": 0.009035207888790999,
"val_accuracy_relaxed_mean": 0.8866003916678595,
"val_accuracy_relaxed_std": 0.009035207888790999,

"test_loss_mean": 1.1702293577302296,
"test_loss_std": 0.09078819056921608,
"test_accuracy_mean": 0.5968146313253844,
"test_accuracy_std": 0.03471874719461086,
"test_accuracy_relaxed_mean": 0.5968146313253844,
"test_accuracy_relaxed_std": 0.03471874719461086
```

#### Not pre-trained

```
"train_loss_mean": 0.2061181211748038,
"train_loss_std": 0.011281717093875578,
"train_accuracy_mean": 0.94325455176554,
"train_accuracy_std": 0.004398384245200454,
"train_accuracy_relaxed_mean": 0.94325455176554,
"train_accuracy_relaxed_std": 0.004398384245200454,

"val_loss_mean": 0.3691450971710101,
"val_loss_std": 0.028719334828979818,
"val_accuracy_mean": 0.8801897830465631,
"val_accuracy_std": 0.008885911812571767,
"val_accuracy_relaxed_mean": 0.8801897830465631,
"val_accuracy_relaxed_std": 0.008885911812571767,

"test_loss_mean": 1.3502018664400444,
"test_loss_std": 0.1500275279723875,
"test_accuracy_mean": 0.5427828841087027,
"test_accuracy_std": 0.04030558887794543,
"test_accuracy_relaxed_mean": 0.5427828841087027,
"test_accuracy_relaxed_std": 0.04030558887794543
```

### Age (mse) LinearBounded

**GARBAGE**

### Age (mse) SigmoidBounded

#### Pre-trained

foo


#### Not pre-trained

foo

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