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
        0.05
    ],
    "num_residuals_per_block": [
        0,
        1,
        2,
        3,
        4,
        5
    ],
    "num_blocks": [
        0,
        1,
        2,
        3,
        4,
        5
    ],
    "batch_size": [
        128,
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
2021-08-04 12:55:52,593 INFO tune.py:549 -- Total run time: 12944.63 seconds (12944.05 seconds for the tuning loop).
Best trial config: OrderedDict([('criterion', 'cse'), ('gender_or_age', 'gender'), ('add_residual', True), ('add_IC', True), ('dropout', 0.05), ('num_residuals_per_block', 1), ('num_blocks', 4), ('batch_size', 512), ('lr', 0.0243983281217744), ('weight_decay', 0.004735384181058588), ('gamma', 0.06620420614596553), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 2), ('validation_split', 0.1)])
Best trial final validation loss: 0.4430247858548776
Best trial final validation accuracy: 0.8243063402385437
```
* time elapsed: 1 hour 10 mins

### `training.py` with `"add_residual": true, "add_IC": true`

#### imdb_wiki
```
2021-08-06 11:06:48,936 - trainer - INFO -     epoch          : 3               
2021-08-06 11:06:48,936 - trainer - INFO -     loss           : 0.4266238472346062
2021-08-06 11:06:48,936 - trainer - INFO -     accuracy       : 0.8299274559694941
2021-08-06 11:06:48,936 - trainer - INFO -     val_loss       : 0.44345825299238545
2021-08-06 11:06:48,936 - trainer - INFO -     val_accuracy   : 0.823281714583733
2021-08-06 11:06:48,963 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_110516/checkpoint-epoch3.pth ...
2021-08-06 11:06:48,998 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 4 mins

#### imdb_wiki_adience

```
2021-08-06 11:15:24,279 - trainer - INFO -     epoch          : 3               
2021-08-06 11:15:24,280 - trainer - INFO -     loss           : 0.41694879128244766
2021-08-06 11:15:24,280 - trainer - INFO -     accuracy       : 0.8351386157660738
2021-08-06 11:15:24,280 - trainer - INFO -     val_loss       : 0.428269509498666
2021-08-06 11:15:24,280 - trainer - INFO -     val_accuracy   : 0.8332944570016821
2021-08-06 11:15:24,305 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_111344/checkpoint-epoch3.pth ...
2021-08-06 11:15:24,329 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed:

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.0649652845712542,                                  
"train_loss_std": 0.007431198822804523,                                 
"train_accuracy_mean": 0.9799522909337441,                              
"train_accuracy_std": 0.002990603883655691,                             
"val_loss_mean": 0.1180148920946893,                                    
"val_loss_std": 0.015007247824669136,                                   
"val_accuracy_mean": 0.9638793030305295,                                
"val_accuracy_std": 0.006335605671980757,                               
"test_loss_mean": 0.43533751480107485,                                  
"test_loss_std": 0.0778508579963934,                                    
"test_accuracy_mean": 0.8406407172280743,                               
"test_accuracy_std": 0.027314883353706715
```
* time elapsed: 13 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.033158706331651626,                                
"train_loss_std": 0.0020978169676838273,                                
"train_accuracy_mean": 0.9916501015955475,                              
"train_accuracy_std": 0.0008641596949750719,                            
"val_loss_mean": 0.08097766418782043,                                   
"val_loss_std": 0.01669467976343782,                                    
"val_accuracy_mean": 0.9771785595684266,                                
"val_accuracy_std": 0.004885199751184193,                               
"test_loss_mean": 0.31197317347965897,                                  
"test_loss_std": 0.0603604492284769,                                    
"test_accuracy_mean": 0.8887175397758311,                               
"test_accuracy_std": 0.025486444846123855  
```
* time elapsed: 14 mins

### `training.py` with `"add_residual": true, "add_IC": false`

`"lr": 0.00243983281217744`


#### imdb_wiki

```
2021-08-06 12:13:51,850 - trainer - INFO -     epoch          : 2               
2021-08-06 12:13:51,850 - trainer - INFO -     loss           : 0.4259779716014182
2021-08-06 12:13:51,850 - trainer - INFO -     accuracy       : 0.8296814132283551
2021-08-06 12:13:51,850 - trainer - INFO -     val_loss       : 0.44093732115549916
2021-08-06 12:13:51,851 - trainer - INFO -     val_accuracy   : 0.8238379661823007
2021-08-06 12:13:51,867 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_121255/checkpoint-epoch2.pth ...
2021-08-06 12:13:51,888 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 3 mins

#### imdb_wiki_adience

```
2021-08-06 12:26:57,911 - trainer - INFO -     epoch          : 2               
2021-08-06 12:26:57,911 - trainer - INFO -     loss           : 0.4146639653742721
2021-08-06 12:26:57,911 - trainer - INFO -     accuracy       : 0.8353951137140903
2021-08-06 12:26:57,911 - trainer - INFO -     val_loss       : 0.42659280540012734
2021-08-06 12:26:57,911 - trainer - INFO -     val_accuracy   : 0.832997135197645
2021-08-06 12:26:57,926 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_122558/checkpoint-epoch2.pth ...
2021-08-06 12:26:57,944 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 3 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.18788806400555194,                                 
"train_loss_std": 0.06184505833485614,                                  
"train_accuracy_mean": 0.9425203519974567,                              
"train_accuracy_std": 0.010204944858290453,                             
"val_loss_mean": 0.20671770950903845,                                   
"val_loss_std": 0.062320597132081997,                                   
"val_accuracy_mean": 0.9335705452192901,                                
"val_accuracy_std": 0.012128948550586624,                               
"test_loss_mean": 0.46219740461201525,                                  
"test_loss_std": 0.09135287489074899,                                   
"test_accuracy_mean": 0.8206668461496278,                               
"test_accuracy_std": 0.03890423963959933
```
* time elapsed: 12 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.04375674086876357,                                 
"train_loss_std": 0.0036730700854976346,                                
"train_accuracy_mean": 0.9880428284821821,                              
"train_accuracy_std": 0.001052018075001737,                             
"val_loss_mean": 0.08951315115628447,                                   
"val_loss_std": 0.014845133446288579,                                   
"val_accuracy_mean": 0.9742719259125172,                                
"val_accuracy_std": 0.004708878367366926,                               
"test_loss_mean": 0.2679020066129207,                                   
"test_loss_std": 0.06755369249286816,                                   
"test_accuracy_mean": 0.9066379871650122,                               
"test_accuracy_std": 0.024821142870521833 
```
* time elapsed: 11 mins

### `training.py` with `"add_residual": false, "add_IC": true`

#### imdb_wiki

```
2021-08-06 13:46:30,414 - trainer - INFO -     epoch          : 5               
2021-08-06 13:46:30,414 - trainer - INFO -     loss           : 0.43179707766939673
2021-08-06 13:46:30,414 - trainer - INFO -     accuracy       : 0.8288361235323164
2021-08-06 13:46:30,414 - trainer - INFO -     val_loss       : 0.4475623296621518
2021-08-06 13:46:30,414 - trainer - INFO -     val_accuracy   : 0.8218971426601765
2021-08-06 13:46:30,441 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_134401/checkpoint-epoch5.pth ...
2021-08-06 13:46:30,469 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 5 mins

#### imdb_wiki_adience

```
2021-08-06 13:52:03,544 - trainer - INFO -     epoch          : 3               
2021-08-06 13:52:03,544 - trainer - INFO -     loss           : 0.422626408074362
2021-08-06 13:52:03,544 - trainer - INFO -     accuracy       : 0.8337439081737346
2021-08-06 13:52:03,544 - trainer - INFO -     val_loss       : 0.42989886279513195
2021-08-06 13:52:03,544 - trainer - INFO -     val_accuracy   : 0.8325519738225399
2021-08-06 13:52:03,565 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_135025/checkpoint-epoch3.pth ...
2021-08-06 13:52:03,592 - trainer - INFO - Saving current best: model_best.pth ...

```
* time elapsed: 4 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.06655765519188118,                                 
"train_loss_std": 0.006702719608215163,                                 
"train_accuracy_mean": 0.9804778616256568,                              
"train_accuracy_std": 0.0021058392205311202,                            
"val_loss_mean": 0.11757890435242942,                                   
"val_loss_std": 0.013003750804260849,                                   
"val_accuracy_mean": 0.961351510139506,                                 
"val_accuracy_std": 0.005179068152188759,                               
"test_loss_mean": 0.42171437494071085,                                  
"test_loss_std": 0.086491244208325,                                     
"test_accuracy_mean": 0.8339792004421082,                               
"test_accuracy_std": 0.03452867618194227   
```
* time elapsed: 14 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.11101154144799123,                                 
"train_loss_std": 0.005004350733177308,                                 
"train_accuracy_mean": 0.9707143079505883,                              
"train_accuracy_std": 0.002071674276009204,                             
"val_loss_mean": 0.13887457862876412,                                   
"val_loss_std": 0.012755607073907185,                                   
"val_accuracy_mean": 0.9596123594913923,                                
"val_accuracy_std": 0.005860486067280916,                               
"test_loss_mean": 0.2833021193040449,                                   
"test_loss_std": 0.03377920896961122,                                   
"test_accuracy_mean": 0.8936712563920685,                               
"test_accuracy_std": 0.01711819441026738   
```
* time elapsed: 14 mins

### `training.py` with `"add_residual": false, "add_IC": false`

`"lr": 0.00243983281217744`

#### imdb_wiki

```
2021-08-06 14:30:11,650 - trainer - INFO -     epoch          : 5               
2021-08-06 14:30:11,651 - trainer - INFO -     loss           : 0.43412846705202707
2021-08-06 14:30:11,651 - trainer - INFO -     accuracy       : 0.831881438261275
2021-08-06 14:30:11,651 - trainer - INFO -     val_loss       : 0.4526786227256824
2021-08-06 14:30:11,651 - trainer - INFO -     val_accuracy   : 0.8221795146548693
2021-08-06 14:30:11,668 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_142755/checkpoint-epoch5.pth ...
2021-08-06 14:30:11,688 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 4 mins

#### imdb_wiki_adience

```
2021-08-06 14:34:59,110 - trainer - INFO -     epoch          : 3               
2021-08-06 14:34:59,110 - trainer - INFO -     loss           : 0.42599035218541503
2021-08-06 14:34:59,110 - trainer - INFO -     accuracy       : 0.8366615723324213
2021-08-06 14:34:59,110 - trainer - INFO -     val_loss       : 0.43823006276677295
2021-08-06 14:34:59,110 - trainer - INFO -     val_accuracy   : 0.8326275362699749
2021-08-06 14:34:59,126 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0806_143330/checkpoint-epoch3.pth ...
2021-08-06 14:34:59,145 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 4 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.6140529103362632,                                  
"train_loss_std": 0.12541141006833903,                                  
"train_accuracy_mean": 0.6118589451562847,                              
"train_accuracy_std": 0.15678530500522103,                              
"val_loss_mean": 0.6144026318646061,                                    
"val_loss_std": 0.12424028169439316,                                    
"val_accuracy_mean": 0.615122525540286,                                 
"val_accuracy_std": 0.15567189629188932,                                
"test_loss_mean": 0.6580882377978682,                                   
"test_loss_std": 0.05811996079536644,                                   
"test_accuracy_mean": 0.5847169636316493,                               
"test_accuracy_std": 0.10150374884096376 
```
* time elapsed: 10 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.06325021642164244,                                 
"train_loss_std": 0.003312433340763357,                                 
"train_accuracy_mean": 0.983338345605302,                               
"train_accuracy_std": 0.001214113184523354,                             
"val_loss_mean": 0.10405783404413019,                                   
"val_loss_std": 0.017748250338883425,                                   
"val_accuracy_mean": 0.9693497658611844,                                
"val_accuracy_std": 0.006105270371733531,                               
"test_loss_mean": 0.2835497287372759,                                   
"test_loss_std": 0.06590264803631932,                                   
"test_accuracy_mean": 0.9033931761843292,                               
"test_accuracy_std": 0.02273976694274134 
```
* time elapsed: 11 mins


## Age, 8 classes

### `hp-tuning.py` with `hp-tuning.json` (below)
```
{
    "criterion": "cse",
    "gender_or_age": "age",
    "add_residual": true,
    "add_IC": true,
    "dropout": [
        0.05
    ],
    "num_residuals_per_block": [
        0,
        1,
        2,
        3,
        4,
        5
    ],
    "num_blocks": [
        0,
        1,
        2,
        3,
        4,
        5
    ],
    "batch_size": [
        128,
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
2021-08-04 13:08:17,614 INFO tune.py:549 -- Total run time: 13657.86 seconds (13657.57 seconds for the tuning loop).
Best trial config: OrderedDict([('criterion', 'cse'), ('gender_or_age', 'age'), ('add_residual', True), ('add_IC', True), ('dropout', 0.05), ('num_residuals_per_block', 1), ('num_blocks', 2), ('batch_size', 128), ('lr', 0.0011670137678831), ('weight_decay', 0.00012337159555124116), ('gamma', 0.3947943239814273), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 8), ('validation_split', 0.1)])
Best trial final validation loss: 1.0595833214047627
Best trial final validation accuracy: 0.6075078468298807
```
* time elapsed: 1 hour 10 mins

### `training.py` with `"add_residual": true, "add_IC": true`

#### imdb_wiki

```
2021-08-04 23:51:11,917 - trainer - INFO -     epoch          : 3               
2021-08-04 23:51:11,917 - trainer - INFO -     loss           : 0.9765892534416005
2021-08-04 23:51:11,917 - trainer - INFO -     accuracy       : 0.6359168600499822
2021-08-04 23:51:11,917 - trainer - INFO -     val_loss       : 1.051012284289568
2021-08-04 23:51:11,917 - trainer - INFO -     val_accuracy   : 0.603936887254902
2021-08-04 23:51:11,935 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_234845/checkpoint-epoch3.pth ...
2021-08-04 23:51:11,955 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 7 mins

#### imdb_wiki_adience

```
2021-08-04 23:59:18,601 - trainer - INFO -     epoch          : 3               
2021-08-04 23:59:18,601 - trainer - INFO -     loss           : 0.9574498667450891
2021-08-04 23:59:18,601 - trainer - INFO -     accuracy       : 0.6445031667237248
2021-08-04 23:59:18,601 - trainer - INFO -     val_loss       : 1.0295328624431903
2021-08-04 23:59:18,602 - trainer - INFO -     val_accuracy   : 0.6133711870026525
2021-08-04 23:59:18,616 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_235657/checkpoint-epoch3.pth ...
2021-08-04 23:59:18,631 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.014739496503208493,                                
"train_loss_std": 0.007489801629906269,                                 
"train_accuracy_mean": 0.9973739930320203,                              
"train_accuracy_std": 0.001717120776471522,                             
"val_loss_mean": 0.2298922029024548,                                    
"val_loss_std": 0.02728060807848834,                                    
"val_accuracy_mean": 0.9342401796563783,                                
"val_accuracy_std": 0.006426674103476173,                               
"test_loss_mean": 1.6408743663705543,                                   
"test_loss_std": 0.18892355076447812,                                   
"test_accuracy_mean": 0.5481515230237024,                               
"test_accuracy_std": 0.03905170389565693 
```
* time elapsed: 13 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.044190126297083084,                                
"train_loss_std": 0.011052668615025676,                                 
"train_accuracy_mean": 0.9906382887582561,                              
"train_accuracy_std": 0.00273898634161635,                              
"val_loss_mean": 0.24981928213786683,                                   
"val_loss_std": 0.02736085388970065,                                    
"val_accuracy_mean": 0.923681059090572,                                 
"val_accuracy_std": 0.0075879276537078235,                              
"test_loss_mean": 1.3566968060235158,                                   
"test_loss_std": 0.10202072351624172,                                   
"test_accuracy_mean": 0.6086133596326488,                               
"test_accuracy_std": 0.028311297707050202 
```
* time elapsed: 12 mins

### `training.py` with `"add_residual": true, "add_IC": false`

#### imdb_wiki

```
2021-08-05 04:21:30,435 - trainer - INFO -     epoch          : 3               
2021-08-05 04:21:30,435 - trainer - INFO -     loss           : 0.9376144870737968
2021-08-05 04:21:30,435 - trainer - INFO -     accuracy       : 0.658817764260017
2021-08-05 04:21:30,435 - trainer - INFO -     val_loss       : 1.0599473462654994
2021-08-05 04:21:30,435 - trainer - INFO -     val_accuracy   : 0.6045010251696833
2021-08-05 04:21:30,450 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0805_041929/checkpoint-epoch3.pth ...
2021-08-05 04:21:30,468 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 5 mins

#### imdb_wiki_adience

```
2021-08-05 13:02:57,074 - trainer - INFO -     epoch          : 3               
2021-08-05 13:02:57,074 - trainer - INFO -     loss           : 0.9191782133949002
2021-08-05 13:02:57,074 - trainer - INFO -     accuracy       : 0.6665659234851078
2021-08-05 13:02:57,074 - trainer - INFO -     val_loss       : 1.0420152730208176
2021-08-05 13:02:57,074 - trainer - INFO -     val_accuracy   : 0.6092125331564987
2021-08-05 13:02:57,090 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0805_125918/checkpoint-epoch3.pth ...
2021-08-05 13:02:57,106 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 7 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.2187553222690939,                                  
"train_loss_std": 0.03288057299008463,                                  
"train_accuracy_mean": 0.9287985040846343,                              
"train_accuracy_std": 0.013761697719029435,                             
"val_loss_mean": 0.4496081069286983,                                    
"val_loss_std": 0.0480034441369954,                                     
"val_accuracy_mean": 0.8596223743640705,                                
"val_accuracy_std": 0.012556593353774586,                               
"test_loss_mean": 1.559061159923447,                                    
"test_loss_std": 0.1445027048988948,                                    
"test_accuracy_mean": 0.5332253618829352,                               
"test_accuracy_std": 0.040315399034449785 
```
* time elapsed: 13 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.09039198173744177,                                 
"train_loss_std": 0.020213096907770824,                                 
"train_accuracy_mean": 0.9798840415092185,                              
"train_accuracy_std": 0.005269091184282561,                             
"val_loss_mean": 0.313296518862654,                                     
"val_loss_std": 0.021045847983206235,                                   
"val_accuracy_mean": 0.9012967936424705,                                
"val_accuracy_std": 0.008174806206558402,                               
"test_loss_mean": 1.340635406269181,                                    
"test_loss_std": 0.09003140226961548,                                   
"test_accuracy_mean": 0.6032404380342925,                               
"test_accuracy_std": 0.03081564764232662  
```
* time elapsed: 9 mins

### `training.py` with `"add_residual": false, "add_IC": true`

#### imdb_wiki

```
2021-08-05 15:30:54,761 - trainer - INFO -     epoch          : 3               
2021-08-05 15:30:54,762 - trainer - INFO -     loss           : 0.9870204366730265
2021-08-05 15:30:54,762 - trainer - INFO -     accuracy       : 0.6341343579216213
2021-08-05 15:30:54,762 - trainer - INFO -     val_loss       : 1.0532616212581978
2021-08-05 15:30:54,762 - trainer - INFO -     val_accuracy   : 0.6026848840497738
2021-08-05 15:30:54,780 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0805_152838/checkpoint-epoch3.pth ...
2021-08-05 15:30:54,802 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### imdb_wiki_adience

```
2021-08-05 15:53:41,049 - trainer - INFO -     epoch          : 3               
2021-08-05 15:53:41,049 - trainer - INFO -     loss           : 0.9665995579147861
2021-08-05 15:53:41,049 - trainer - INFO -     accuracy       : 0.6420238146182814
2021-08-05 15:53:41,050 - trainer - INFO -     val_loss       : 1.0365553855895997
2021-08-05 15:53:41,050 - trainer - INFO -     val_accuracy   : 0.6093658819628648
2021-08-05 15:53:41,065 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0805_155112/checkpoint-epoch3.pth ...
2021-08-05 15:53:41,085 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.015749924983809914,                                
"train_loss_std": 0.003948447945478645,                                 
"train_accuracy_mean": 0.9971000708507902,                              
"train_accuracy_std": 0.0008076148841114056,                            
"val_loss_mean": 0.23301234392944054,                                   
"val_loss_std": 0.026114256791936914,                                   
"val_accuracy_mean": 0.933097924513062,                                 
"val_accuracy_std": 0.00789377556150749,                                
"test_loss_mean": 1.6680036840452248,                                   
"test_loss_std": 0.13916720759903736,                                   
"test_accuracy_mean": 0.5394806824328668,                               
"test_accuracy_std": 0.03735867803724017 
```
* time elapsed: 22 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.05376974607977008,                                 
"train_loss_std": 0.018095553531624658,                                 
"train_accuracy_mean": 0.987268581428263,                               
"train_accuracy_std": 0.004798754865327981,                             
"val_loss_mean": 0.25993021216371975,                                   
"val_loss_std": 0.022601937888301005,                                   
"val_accuracy_mean": 0.9209645748315722,                                
"val_accuracy_std": 0.00825128876313552,                                
"test_loss_mean": 1.3822360941715122,                                   
"test_loss_std": 0.09632680122948459,                                   
"test_accuracy_mean": 0.6046470200696574,                               
"test_accuracy_std": 0.026999898209333337  
```
* time elapsed: 12 mins

### `training.py` with `"add_residual": false, "add_IC": false`


#### imdb_wiki

```
2021-08-05 16:41:58,588 - trainer - INFO -     epoch          : 3               
2021-08-05 16:41:58,588 - trainer - INFO -     loss           : 0.9487060637867311
2021-08-05 16:41:58,588 - trainer - INFO -     accuracy       : 0.6548966028616153
2021-08-05 16:41:58,589 - trainer - INFO -     val_loss       : 1.0638830422973022
2021-08-05 16:41:58,589 - trainer - INFO -     val_accuracy   : 0.6047013456825038
2021-08-05 16:41:58,605 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0805_163951/checkpoint-epoch3.pth ...
2021-08-05 16:41:58,620 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### imdb_wiki_adience

```
2021-08-05 16:50:45,553 - trainer - INFO -     epoch          : 3               
2021-08-05 16:50:45,553 - trainer - INFO -     loss           : 0.9307046102685742
2021-08-05 16:50:45,553 - trainer - INFO -     accuracy       : 0.6631665097569326
2021-08-05 16:50:45,553 - trainer - INFO -     val_loss       : 1.0494548025498023
2021-08-05 16:50:45,553 - trainer - INFO -     val_accuracy   : 0.6076881631299734
2021-08-05 16:50:45,569 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0805_164830/checkpoint-epoch3.pth ...
2021-08-05 16:50:45,589 - trainer - INFO - Saving current best: model_best.pth ...

```
* time elapsed: 6 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.42018930540817506,                                 
"train_loss_std": 0.07968775186548589,                                  
"train_accuracy_mean": 0.8358889114293895,                              
"train_accuracy_std": 0.04280607983689087,                              
"val_loss_mean": 0.7315186434066174,                                    
"val_loss_std": 0.09693372029217172,                                    
"val_accuracy_mean": 0.7547668405181192,                                
"val_accuracy_std": 0.043093439353469855,                               
"test_loss_mean": 2.071858689348585,                                    
"test_loss_std": 0.24537985674762677,                                   
"test_accuracy_mean": 0.40679957475404477,                              
"test_accuracy_std": 0.05287554302722316 
```
* time elapsed: 12 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.1052458881772117,                                  
"train_loss_std": 0.012792726572766104,                                 
"train_accuracy_mean": 0.9750527317182842,                              
"train_accuracy_std": 0.0037080993861352074,                            
"val_loss_mean": 0.33074651501030894,                                   
"val_loss_std": 0.025214695767740155,                                   
"val_accuracy_mean": 0.8932273361153038,                                
"val_accuracy_std": 0.008698612620235711,                               
"test_loss_mean": 1.356803093728347,                                    
"test_loss_std": 0.1310938838013513,                                    
"test_accuracy_mean": 0.5936561489498906,                               
"test_accuracy_std": 0.03572556187687143  
```
* time elapsed: 9 mins

## Age, 101 classes

### `hp-tuning.py` with `hp-tuning.json` (below)
```
{
    "criterion": "cse",
    "gender_or_age": "age",
    "add_residual": true,
    "add_IC": true,
    "dropout": [
        0.05
    ],
    "num_residuals_per_block": [
        0,
        1,
        2,
        3,
        4,
        5
    ],
    "num_blocks": [
        0,
        1,
        2,
        3,
        4,
        5
    ],
    "batch_size": [
        128,
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
    "num_classes": 101,
    "validation_split": 0.1
}
```

```
2021-08-04 12:42:32,529 INFO tune.py:549 -- Total run time: 12076.78 seconds (12076.05 seconds for the tuning loop).
Best trial config: OrderedDict([('criterion', 'cse'), ('gender_or_age', 'age'), ('add_residual', True), ('add_IC', True), ('dropout', 0.05), ('num_residuals_per_block', 1), ('num_blocks', 2), ('batch_size', 128), ('lr', 0.0011670137678831), ('weight_decay', 0.00012337159555124116), ('gamma', 0.3947943239814273), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 101), ('validation_split', 0.1)])
Best trial final validation loss: 3.287252651575284
Best trial final validation accuracy: 0.1598242310106717
```
* time elapsed: 1 hour 10 mins

### `training.py` with `"add_residual": true, "add_IC": true`

#### imdb_wiki

```
2021-08-04 12:59:58,923 - trainer - INFO -     epoch          : 7               
2021-08-04 12:59:58,923 - trainer - INFO -     loss           : 3.0200273145398513
2021-08-04 12:59:58,924 - trainer - INFO -     accuracy       : 0.21119861038640048
2021-08-04 12:59:58,924 - trainer - INFO -     val_loss       : 3.2814942552493167
2021-08-04 12:59:58,924 - trainer - INFO -     val_accuracy   : 0.1608706282993967
2021-08-04 12:59:58,940 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_125337/checkpoint-epoch7.pth ...
2021-08-04 12:59:58,959 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 10 mins

#### imdb_wiki_adience

```
2021-08-04 13:10:44,975 - trainer - INFO -     epoch          : 5               
2021-08-04 13:10:44,975 - trainer - INFO -     loss           : 2.9644558335036213
2021-08-04 13:10:44,975 - trainer - INFO -     accuracy       : 0.22686472954467649
2021-08-04 13:10:44,976 - trainer - INFO -     val_loss       : 3.196785119130061
2021-08-04 13:10:44,976 - trainer - INFO -     val_accuracy   : 0.18462201591511937
2021-08-04 13:10:44,993 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_130630/checkpoint-epoch5.pth ...
2021-08-04 13:10:45,013 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 7 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.018803563352668903,                                
"train_loss_std": 0.003981359137520785,                                 
"train_accuracy_mean": 0.9968163529584783,                              
"train_accuracy_std": 0.0007963574511976345,                            
"val_loss_mean": 0.23540298709051374,                                   
"val_loss_std": 0.02706932878500526,                                    
"val_accuracy_mean": 0.9343246823460251,                                
"val_accuracy_std": 0.007345127657187889,                               
"test_loss_mean": 1.6534216372526924,                                   
"test_loss_std": 0.17353274901325103,                                   
"test_accuracy_mean": 0.5444984608803018,                               
"test_accuracy_std": 0.03969608603733163
```
* time elapsed: 13 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.04545511317544952,                                 
"train_loss_std": 0.011582151765869825,                                 
"train_accuracy_mean": 0.9912913328381187,                              
"train_accuracy_std": 0.0027117227668752996,                            
"val_loss_mean": 0.26228065723772653,                                   
"val_loss_std": 0.027238944790212685,                                   
"val_accuracy_mean": 0.919461702405183,                                 
"val_accuracy_std": 0.008025233795177144,                               
"test_loss_mean": 1.3636066654437713,                                   
"test_loss_std": 0.06940663370589772,                                   
"test_accuracy_mean": 0.6005364514701527,                               
"test_accuracy_std": 0.023349385815310295 
```
* time elapsed: 14 mins

### `training.py` with `"add_residual": true, "add_IC": false`

#### imdb_wiki

```
2021-08-04 14:09:31,159 - trainer - INFO -     epoch          : 4               
2021-08-04 14:09:31,159 - trainer - INFO -     loss           : 3.161506777276145
2021-08-04 14:09:31,159 - trainer - INFO -     accuracy       : 0.14746343167275425
2021-08-04 14:09:31,159 - trainer - INFO -     val_loss       : 3.3710454243880053
2021-08-04 14:09:31,160 - trainer - INFO -     val_accuracy   : 0.11807863169306185
2021-08-04 14:09:31,176 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_140648/checkpoint-epoch4.pth ...
2021-08-04 14:09:31,194 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### imdb_wiki_adience

```
2021-08-04 14:16:42,582 - trainer - INFO -     epoch          : 3               
2021-08-04 14:16:42,583 - trainer - INFO -     loss           : 3.12881465163879
2021-08-04 14:16:42,583 - trainer - INFO -     accuracy       : 0.16165803663129066
2021-08-04 14:16:42,583 - trainer - INFO -     val_loss       : 3.2822727467463566
2021-08-04 14:16:42,583 - trainer - INFO -     val_accuracy   : 0.1421991047745358
2021-08-04 14:16:42,595 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_141429/checkpoint-epoch3.pth ...
2021-08-04 14:16:42,611 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 5 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.453935819426141,                                   
"train_loss_std": 0.047157416142341924,                                 
"train_accuracy_mean": 0.8339816696791367,                              
"train_accuracy_std": 0.019246473110951536,                             
"val_loss_mean": 0.5811139679108489,                                    
"val_loss_std": 0.04081904610737358,                                    
"val_accuracy_mean": 0.7932559053275473,                                
"val_accuracy_std": 0.018348902420289123,                               
"test_loss_mean": 1.3348493398058838,                                   
"test_loss_std": 0.14247212927363231,                                   
"test_accuracy_mean": 0.5329012058797243,                               
"test_accuracy_std": 0.03716699659878682 
```
* time elapsed: 18 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.10999468074281796,                                 
"train_loss_std": 0.021556805519474653,                                 
"train_accuracy_mean": 0.9726441508684591,                              
"train_accuracy_std": 0.006406822795072672,                             
"val_loss_mean": 0.34819507628122076,                                   
"val_loss_std": 0.02380311932283215,                                    
"val_accuracy_mean": 0.8930195217194301,                                
"val_accuracy_std": 0.0065059300009134,                                 
"test_loss_mean": 1.3879593566341297,                                   
"test_loss_std": 0.07896728725735712,                                   
"test_accuracy_mean": 0.6030450285446743,                               
"test_accuracy_std": 0.021638053524286733 
```
* time elapsed: 10 mins

### `training.py` with `"add_residual": false, "add_IC": true`

#### imdb_wiki

```
2021-08-04 15:44:57,658 - trainer - INFO -     epoch          : 8               
2021-08-04 15:44:57,658 - trainer - INFO -     loss           : 3.0857693725124253
2021-08-04 15:44:57,658 - trainer - INFO -     accuracy       : 0.1863420601845495
2021-08-04 15:44:57,658 - trainer - INFO -     val_loss       : 3.3059655275100317
2021-08-04 15:44:57,658 - trainer - INFO -     val_accuracy   : 0.14763769089366516
2021-08-04 15:44:57,679 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_153854/checkpoint-epoch8.pth ...
2021-08-04 15:44:57,702 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 10 mins

#### imdb_wiki_adience

```
2021-08-04 16:03:16,399 - trainer - INFO -     epoch          : 16              
2021-08-04 16:03:16,399 - trainer - INFO -     loss           : 3.0069671828874944
2021-08-04 16:03:16,399 - trainer - INFO -     accuracy       : 0.20983556573091408
2021-08-04 16:03:16,399 - trainer - INFO -     val_loss       : 3.224575438132653
2021-08-04 16:03:16,399 - trainer - INFO -     val_accuracy   : 0.1721551724137931
2021-08-04 16:03:16,418 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_155115/checkpoint-epoch16.pth ...
2021-08-04 16:03:16,439 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 15 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.01984713265981791,                                 
"train_loss_std": 0.0049036925402442125,                                
"train_accuracy_mean": 0.9965252429269149,                              
"train_accuracy_std": 0.0008567829984185799,                            
"val_loss_mean": 0.2336303363418904,                                    
"val_loss_std": 0.027047535744339275,                                   
"val_accuracy_mean": 0.9326447526841612,                                
"val_accuracy_std": 0.008075767818661417,                               
"test_loss_mean": 1.6528112440090832,                                   
"test_loss_std": 0.16714354925780017,                                   
"test_accuracy_mean": 0.5434543605071928,                               
"test_accuracy_std": 0.037108616301812565
```
* time elapsed: 15 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.044960734909356724,                                
"train_loss_std": 0.007755128736973053,                                 
"train_accuracy_mean": 0.990803508707438,                               
"train_accuracy_std": 0.0020723646887524454,                            
"val_loss_mean": 0.26500815759495355,                                   
"val_loss_std": 0.02852086799293112,                                    
"val_accuracy_mean": 0.9188075365091344,                                
"val_accuracy_std": 0.008018510317906943,                               
"test_loss_mean": 1.4405305576859275,                                   
"test_loss_std": 0.08819631245400522,                                   
"test_accuracy_mean": 0.5984918813045172,                               
"test_accuracy_std": 0.02257798885132503    
```
* time elapsed: 13 mins

### `training.py` with `"add_residual": false, "add_IC": false`


#### imdb_wiki

```
2021-08-04 22:38:52,718 - trainer - INFO -     epoch          : 3               
2021-08-04 22:38:52,719 - trainer - INFO -     loss           : 3.277826986922319
2021-08-04 22:38:52,719 - trainer - INFO -     accuracy       : 0.10334714765331064
2021-08-04 22:38:52,719 - trainer - INFO -     val_loss       : 3.410101289168382
2021-08-04 22:38:52,719 - trainer - INFO -     val_accuracy   : 0.086640094739819
2021-08-04 22:38:52,733 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_223640/checkpoint-epoch3.pth ...
2021-08-04 22:38:52,749 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### imdb_wiki_adience

```
2021-08-04 22:57:13,847 - trainer - INFO -     epoch          : 2               
2021-08-04 22:57:13,848 - trainer - INFO -     loss           : 3.326024848522694
2021-08-04 22:57:13,848 - trainer - INFO -     accuracy       : 0.11016400633344745
2021-08-04 22:57:13,848 - trainer - INFO -     val_loss       : 3.3469222017434928
2021-08-04 22:57:13,848 - trainer - INFO -     val_accuracy   : 0.11118368700265252
2021-08-04 22:57:13,860 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0804_225547/checkpoint-epoch2.pth ...
2021-08-04 22:57:13,875 - trainer - INFO - Saving current best: model_best.pth ...
```
* time elapsed: 6 mins

#### cross-val on adience, from random initialization

```
"train_loss_mean": 0.8561103206518992,                                  
"train_loss_std": 0.13042864271613142,                                  
"train_accuracy_mean": 0.6270276050850532,                              
"train_accuracy_std": 0.05837942302416236,                              
"val_loss_mean": 0.9392933148207148,                                    
"val_loss_std": 0.11056936853059371,                                    
"val_accuracy_mean": 0.5967305282588963,                                
"val_accuracy_std": 0.051856446770754364,                               
"test_loss_mean": 1.4230983414952312,                                   
"test_loss_std": 0.11275823480829542,                                   
"test_accuracy_mean": 0.4269996969331043,                               
"test_accuracy_std": 0.03880932637987499
```
* time elapsed: 19 mins

#### cross-val on adience, pretrained on imdb_wiki

```
"train_loss_mean": 0.13112277342150466,                                 
"train_loss_std": 0.022106397443660244,                                 
"train_accuracy_mean": 0.9655133550162183,                              
"train_accuracy_std": 0.0068467818244735456,                            
"val_loss_mean": 0.353777743741244,                                     
"val_loss_std": 0.02299664392362703,                                    
"val_accuracy_mean": 0.8889953931138357,                                
"val_accuracy_std": 0.0064907870101817014,                              
"test_loss_mean": 1.388671813502354,                                    
"test_loss_std": 0.1497105826467238,                                    
"test_accuracy_mean": 0.5967172266179805,                               
"test_accuracy_std": 0.03440994418228202 
```
* time elapsed: 9 mins