# Training results

## Gender, 2 classes

### `hp-tuning.py` with `hp-tuning.json` (below)
```
{
    "criterion": "cse",
    "gender_or_age": "gender",
    "add_residual": true,
    "add_IC": true,
    "dropout": [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    ],
    "num_residuals_per_block": [
        0,
        1,
        2,
        3,
        4
    ],
    "num_blocks": [
        0,
        1,
        2,
        3,
        4
    ],
    "batch_size": [
        256,
        512
    ],
    "lr": [
        1e-6,
        1e-1
    ],
    "weight_decay": [
        1e-6,
        1e-1
    ],
    "gamma": [
        1e-6,
        1
    ],
    "data_dir": "/home/tk/repos/age-gender/data",
    "cpus": 8,
    "dataset": "imdb_wiki",
    "num_samples": 100,
    "max_num_epochs": 10,
    "gpus_per_trial": 1,
    "limit_data": null,
    "amp": true,
    "num_classes": 2,
    "validation_split": 0.1
}
```
```
2021-08-03 12:22:12,370 INFO tune.py:549 -- Total run time: 5064.50 seconds (5063.90 seconds for the tuning loop).
Best trial config: OrderedDict([('criterion', 'cse'), ('gender_or_age', 'gender'), ('add_residual', True), ('add_IC', True), ('dropout', 0.1), ('num_residuals_per_block', 2), ('num_blocks', 4), ('batch_size', 256), ('lr', 0.03943508299495083), ('weight_decay', 2.5926393015905714e-06), ('gamma', 0.18195706189458422), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 2), ('validation_split', 0.1)])
Best trial final validation loss: 0.4510381668806076
Best trial final validation accuracy: 0.8213935969868174
```
* time elapsed: 1 hour 22 minutes

### `training.py` with `"add_residual": true, "add_IC": true`

#### imdb_wiki
```
2021-08-03 16:12:53,643 - trainer - INFO -     epoch          : 10              
2021-08-03 16:12:53,643 - trainer - INFO -     loss           : 0.43562397133189384
2021-08-03 16:12:53,643 - trainer - INFO -     accuracy       : 0.8269820578021194
2021-08-03 16:12:53,643 - trainer - INFO -     val_loss       : 0.4500062224956659
2021-08-03 16:12:53,643 - trainer - INFO -     val_accuracy   : 0.8211075652077807
2021-08-03 16:12:53,677 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_160516/checkpoint-epoch10.pth ...
2021-08-03 16:12:53,714 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 11 mins

#### imdb_wiki_adience

```
2021-08-03 16:24:36,347 - trainer - INFO -     epoch          : 8               
2021-08-03 16:24:36,347 - trainer - INFO -     loss           : 0.4273104392078786
2021-08-03 16:24:36,348 - trainer - INFO -     accuracy       : 0.8324831023271732
2021-08-03 16:24:36,348 - trainer - INFO -     val_loss       : 0.43388564253877276
2021-08-03 16:24:36,348 - trainer - INFO -     val_accuracy   : 0.8313428839644594
2021-08-03 16:24:36,387 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_161822/checkpoint-epoch8.pth ...
2021-08-03 16:24:36,426 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 10 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.07014234351000409,                                 
"train_loss_std": 0.007419605746854006,                                 
"train_accuracy_mean": 0.9785880285591491,                              
"train_accuracy_std": 0.00268523516573176,                              
"val_loss_mean": 0.13736265950715068,                                   
"val_loss_std": 0.019768350925692325,                                   
"val_accuracy_mean": 0.9542454167610753,                                
"val_accuracy_std": 0.005536124699840537,                               
"test_loss_mean": 0.4753054151811527,                                   
"test_loss_std": 0.10246762597068003,                                   
"test_accuracy_mean": 0.8168209149712148,                               
"test_accuracy_std": 0.03200967909062663
```
* time elapsed: 15 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.05116955817088978,                                 
"train_loss_std": 0.027093508683019133,                                 
"train_accuracy_mean": 0.992713610982031,                               
"train_accuracy_std": 0.007301648812779891,                             
"val_loss_mean": 0.1018548136591801,                                    
"val_loss_std": 0.03267981903184156,                                    
"val_accuracy_mean": 0.9733501212112605,                                
"val_accuracy_std": 0.010233636484555785,                               
"test_loss_mean": 0.3645594851484624,                                   
"test_loss_std": 0.08713938851396215,                                   
"test_accuracy_mean": 0.8742039684103835,                               
"test_accuracy_std": 0.030556116361374747 
```
* time elapsed: 13 mins

### `training.py` with `"add_residual": true, "add_IC": false`

* ('lr', 0.0003943508299495083)
  
#### imdb_wiki

```
2021-08-03 20:14:27,519 - trainer - INFO -     epoch          : 6               
2021-08-03 20:14:27,520 - trainer - INFO -     loss           : 0.4296978085403524
2021-08-03 20:14:27,520 - trainer - INFO -     accuracy       : 0.8314330766485478
2021-08-03 20:14:27,520 - trainer - INFO -     val_loss       : 0.4511282864289406
2021-08-03 20:14:27,520 - trainer - INFO -     val_accuracy   : 0.8199615937223695
2021-08-03 20:14:27,545 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_201052/checkpoint-epoch6.pth ...
2021-08-03 20:14:27,573 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### imdb_wiki_adience

```
2021-08-03 20:22:36,012 - trainer - INFO -     epoch          : 5               
2021-08-03 20:22:36,012 - trainer - INFO -     loss           : 0.4202028327735624
2021-08-03 20:22:36,012 - trainer - INFO -     accuracy       : 0.8369428045859001
2021-08-03 20:22:36,012 - trainer - INFO -     val_loss       : 0.4355802848660873
2021-08-03 20:22:36,012 - trainer - INFO -     val_accuracy   : 0.8304578419187646
2021-08-03 20:22:36,038 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_201930/checkpoint-epoch5.pth ...
2021-08-03 20:22:36,066 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.2427718426573297,                                  
"train_loss_std": 0.05739613804765838,                                  
"train_accuracy_mean": 0.9150258346057791,                              
"train_accuracy_std": 0.008623799047669793,                             
"val_loss_mean": 0.2569146760947735,                                    
"val_loss_std": 0.05539370526844229,                                    
"val_accuracy_mean": 0.9102971799542968,                                
"val_accuracy_std": 0.009436896248468095,                               
"test_loss_mean": 0.4620544001578747,                                   
"test_loss_std": 0.09472440369651962,                                   
"test_accuracy_mean": 0.8100161602910977,                               
"test_accuracy_std": 0.038296994731876824 
```
* time elapsed: 15 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.06452259697621805,                                 
"train_loss_std": 0.005104663719793042,                                 
"train_accuracy_mean": 0.9826768799552934,                              
"train_accuracy_std": 0.0019551576166415073,                            
"val_loss_mean": 0.12104016323904697,                                   
"val_loss_std": 0.017929299356215567,                                   
"val_accuracy_mean": 0.9656003295661434,                                
"val_accuracy_std": 0.006911336096919621,                               
"test_loss_mean": 0.2997497136718888,                                   
"test_loss_std": 0.04935490312943048,                                   
"test_accuracy_mean": 0.8945160037944951,                               
"test_accuracy_std": 0.01877370748328877
```
* time elapsed: 12 mins

### `training.py` with `"add_residual": false, "add_IC": true`

#### imdb_wiki

```
2021-08-03 21:46:54,969 - trainer - INFO -     epoch          : 10              
2021-08-03 21:46:54,969 - trainer - INFO -     loss           : 0.4508944748981266
2021-08-03 21:46:54,969 - trainer - INFO -     accuracy       : 0.8221083069098996
2021-08-03 21:46:54,969 - trainer - INFO -     val_loss       : 0.45038493837301546
2021-08-03 21:46:54,969 - trainer - INFO -     val_accuracy   : 0.8216776152188329
2021-08-03 21:46:55,003 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_213926/checkpoint-epoch10.pth ...
2021-08-03 21:46:55,039 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 11 mins

#### imdb_wiki_adience

```
2021-08-03 22:01:08,327 - trainer - INFO -     epoch          : 11              
2021-08-03 22:01:08,328 - trainer - INFO -     loss           : 0.43982255434353495
2021-08-03 22:01:08,328 - trainer - INFO -     accuracy       : 0.8283469156399726
2021-08-03 22:01:08,328 - trainer - INFO -     val_loss       : 0.4354807381615317
2021-08-03 22:01:08,328 - trainer - INFO -     val_accuracy   : 0.831406514438333
2021-08-03 22:01:08,362 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_215217/checkpoint-epoch11.pth ...
2021-08-03 22:01:08,404 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 12 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.1872913010730302,                                  
"train_loss_std": 0.02806141880510625,                                  
"train_accuracy_mean": 0.9306303536671053,                              
"train_accuracy_std": 0.01244263662557745,                              
"val_loss_mean": 0.21449545105333423,                                   
"val_loss_std": 0.03229871949268204,                                    
"val_accuracy_mean": 0.9199822739355062,                                
"val_accuracy_std": 0.014667411625760962,                               
"test_loss_mean": 0.4761799967033295,                                   
"test_loss_std": 0.08907125424105215,                                   
"test_accuracy_mean": 0.7889961510473352,                               
"test_accuracy_std": 0.04261486888906439
```
* time elapsed: 16 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.08476811661805007,                                 
"train_loss_std": 0.021271091650433744,                                 
"train_accuracy_mean": 0.9746272717103195,                              
"train_accuracy_std": 0.006716743616546124,                             
"val_loss_mean": 0.11509117217913556,                                   
"val_loss_std": 0.01734963954698643,                                    
"val_accuracy_mean": 0.9627477525201106,                                
"val_accuracy_std": 0.005571161200967543,                               
"test_loss_mean": 0.34043287715068454,                                  
"test_loss_std": 0.086842181381673,                                     
"test_accuracy_mean": 0.8619535349393429,                               
"test_accuracy_std": 0.03815758499281301 
```
* time elapsed: 17 mins

### `training.py` with `"add_residual": false, "add_IC": false`

* ('lr', 0.0003943508299495083)

#### imdb_wiki

```
2021-08-03 19:20:03,976 - trainer - INFO -     epoch          : 10              
2021-08-03 19:20:03,976 - trainer - INFO -     loss           : 0.42602775547438054
2021-08-03 19:20:03,976 - trainer - INFO -     accuracy       : 0.8312675012353814
2021-08-03 19:20:03,976 - trainer - INFO -     val_loss       : 0.44402438211135375
2021-08-03 19:20:03,976 - trainer - INFO -     val_accuracy   : 0.8237058604111406
2021-08-03 19:20:04,010 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_191409/checkpoint-epoch10.pth ...
2021-08-03 19:20:04,048 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 10 mins

#### imdb_wiki_adience

```
2021-08-03 19:29:41,551 - trainer - INFO -     epoch          : 5               
2021-08-03 19:29:41,551 - trainer - INFO -     loss           : 0.4150009904105234
2021-08-03 19:29:41,551 - trainer - INFO -     accuracy       : 0.8357851001026694
2021-08-03 19:29:41,551 - trainer - INFO -     val_loss       : 0.4286481776486145
2021-08-03 19:29:41,551 - trainer - INFO -     val_accuracy   : 0.8327824201396236
2021-08-03 19:29:41,577 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0803_192633/checkpoint-epoch5.pth ...
2021-08-03 19:29:41,613 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.6907346900355976,                                  
"train_loss_std": 0.0013145803689440204,                                
"train_accuracy_mean": 0.5336449299202591,                              
"train_accuracy_std": 0.007950648528364619,                             
"val_loss_mean": 0.6906658022175723,                                    
"val_loss_std": 0.0009101257046015031,                                  
"val_accuracy_mean": 0.5360675608998604,                                
"val_accuracy_std": 0.006592316413961109,                               
"test_loss_mean": 0.6917365946485787,                                   
"test_loss_std": 0.004936763252845446,                                  
"test_accuracy_mean": 0.5358384483326789,                               
"test_accuracy_std": 0.025814188969479  
```
* time elapsed: 11 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.05266124948346498,                                 
"train_loss_std": 0.003660504798093344,                                 
"train_accuracy_mean": 0.9853635179329782,                              
"train_accuracy_std": 0.001421638410017778,                             
"val_loss_mean": 0.10465143326075514,                                   
"val_loss_std": 0.013137950302336387,                                   
"val_accuracy_mean": 0.9697618674917916,                                
"val_accuracy_std": 0.004438335739604548,                               
"test_loss_mean": 0.27407862231075136,                                  
"test_loss_std": 0.07626170155366795,                                   
"test_accuracy_mean": 0.9038577370589581,                               
"test_accuracy_std": 0.02926999036433687 
```
* time elapsed: 12 mins


## Age, 8 classes

### `hp-tuning.py` with `hp-tuning.json` (below)
```
{
    "criterion": "cse",
    "gender_or_age": "age",
    "add_residual": true,
    "add_IC": true,
    "dropout": [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    ],
    "num_residuals_per_block": [
        0,
        1,
        2,
        3,
        4
    ],
    "num_blocks": [
        0,
        1,
        2,
        3,
        4
    ],
    "batch_size": [
        256,
        512
    ],
    "lr": [
        1e-6,
        1e-1
    ],
    "weight_decay": [
        1e-6,
        1e-1
    ],
    "gamma": [
        1e-6,
        1
    ],
    "data_dir": "/home/tk/repos/age-gender/data",
    "cpus": 8,
    "dataset": "imdb_wiki",
    "num_samples": 100,
    "max_num_epochs": 10,
    "gpus_per_trial": 1,
    "limit_data": null,
    "amp": true,
    "num_classes": 8,
    "validation_split": 0.1
}
```

```
2021-08-04 00:09:54,261 INFO tune.py:549 -- Total run time: 5026.78 seconds (5026.51 seconds for the tuning loop).
Best trial config: OrderedDict([('criterion', 'cse'), ('gender_or_age', 'age'), ('add_residual', True), ('add_IC', True), ('dropout', 0.0), ('num_residuals_per_block', 2), ('num_blocks', 3), ('batch_size', 512), ('lr', 0.0030374373651663737), ('weight_decay', 0.00011436840694269902), ('gamma', 0.17615932159571032), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 8), ('validation_split', 0.1)])
Best trial final validation loss: 1.0998443961143494
Best trial final validation accuracy: 0.5889516635279347
```
* time elapsed: 1 hour 23 mins

### `training.py` with `"add_residual": true, "add_IC": true`

#### imdb_wiki

```
2021-08-04 00:20:35,084 - trainer - INFO -     epoch          : 3               
2021-08-04 00:20:35,084 - trainer - INFO -     loss           : 0.9703453184024413
2021-08-04 00:20:35,084 - trainer - INFO -     accuracy       : 0.6441737593273346
2021-08-04 00:20:35,084 - trainer - INFO -     val_loss       : 1.0886048193161304
2021-08-04 00:20:35,084 - trainer - INFO -     val_accuracy   : 0.589514107859358
2021-08-04 00:20:35,114 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_001903/checkpoint-epoch3.pth ...
2021-08-04 00:20:35,146 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 4 mins

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed:

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed:

### `training.py` with `"add_residual": true, "add_IC": false`

#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed:

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed:

### `training.py` with `"add_residual": false, "add_IC": true`

#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed: 

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed: 

### `training.py` with `"add_residual": false, "add_IC": false`


#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed: 

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed: 

## Age, 101 classes

### `hp-tuning.py` with `hp-tuning.json` (below)
```
```

```
```
* time elapsed:

### `training.py` with `"add_residual": true, "add_IC": true`

#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed:

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed:

### `training.py` with `"add_residual": true, "add_IC": false`

#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed:

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed:

### `training.py` with `"add_residual": false, "add_IC": true`

#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed: 

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed: 

### `training.py` with `"add_residual": false, "add_IC": false`


#### imdb_wiki

```
```
* time elapsed:

#### imdb_wiki_adience

```
```
* time elapsed:

#### cross-val on adience, from random initialization

```
```
* time elapsed: 

#### cross-val on adience, pretrained on imdb_wiki

```
```
* time elapsed: 