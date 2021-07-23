# Evaluation results (MLP with IC)

Validation split is always 10%

## Gender (2 classes, cross entropy loss)

### Train on IMDB and WIKI

2021-07-21 11:16:39,825 INFO tune.py:549 -- Total run time: 5317.88 seconds (5317.23 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'gender'), ('only_MLP', False), ('dropout', 0.5), ('num_residuals_per_block', 3), ('num_blocks', 1), ('batch_size', 512), ('lr', 0.010979988817809663), ('weight_decay', 0.001468989807764881), ('gamma', 0.16934155410667961), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 2), ('validation_split', 0.1)])
Best trial final validation loss: 0.44577305018901825
Best trial final validation accuracy: 0.8241807909604519

```
2021-07-21 11:31:20,345 - trainer - INFO -     epoch          : 10
2021-07-21 11:31:20,345 - trainer - INFO -     loss           : 0.4415791288081998
2021-07-21 11:31:20,346 - trainer - INFO -     accuracy       : 0.8253154405108087
2021-07-21 11:31:20,346 - trainer - INFO -     val_loss       : 0.44690019006912524
2021-07-21 11:31:20,346 - trainer - INFO -     val_accuracy   : 0.8229877279957157
2021-07-21 11:31:20,384 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0721_112606/checkpoint-epoch10.pth ...
2021-07-21 11:31:20,422 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on IMDB, WIKI, and Adience

2021-07-21 11:18:37,148 INFO tune.py:549 -- Total run time: 5415.20 seconds (5414.64 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'gender'), ('only_MLP', False), ('dropout', 0.5), ('num_residuals_per_block', 3), ('num_blocks', 1), ('batch_size', 512), ('lr', 0.010979988817809663), ('weight_decay', 0.001468989807764881), ('gamma', 0.16934155410667961), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki_adience'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 2), ('validation_split', 0.1)])
Best trial final validation loss: 0.43183608643892335
Best trial final validation accuracy: 0.8323862268239827

```
2021-07-21 11:31:08,903 - trainer - INFO -     epoch          : 9
2021-07-21 11:31:08,903 - trainer - INFO -     loss           : 0.4347131204181102
2021-07-21 11:31:08,904 - trainer - INFO -     accuracy       : 0.8298884233926128
2021-07-21 11:31:08,904 - trainer - INFO -     val_loss       : 0.43157466155726737
2021-07-21 11:31:08,904 - trainer - INFO -     val_accuracy   : 0.8325798990748529
2021-07-21 11:31:08,942 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0721_112613/checkpoint-epoch9.pth ...
2021-07-21 11:31:09,000 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on Adience from scratch (5 times 5-fold cross validation)

saved/0721_114435_cross-val-results.json
```
"train_loss_mean": 0.10148818105643322,
"train_loss_std": 0.008759538974549577,
"train_accuracy_mean": 0.9649996459765515,
"train_accuracy_std": 0.00356599374993471,
"val_loss_mean": 0.14238248450113583,
"val_loss_std": 0.02108279431430651,
"val_accuracy_mean": 0.9533133750599464,
"val_accuracy_std": 0.0058745037190235235,
"test_loss_mean": 0.41745343956221725,
"test_loss_std": 0.08806107479811218,
"test_accuracy_mean": 0.8338550235483859,
"test_accuracy_std": 0.03694679438854957
```

### Pre-trained on IMDB and WIKI, fine-tune on Adience (5 times 5-fold cross validation)

saved/0721_115511_cross-val-results.json

```
"train_loss_mean": 0.038070759118852555,
"train_loss_std": 0.0023812576659911246,
"train_accuracy_mean": 0.9896795583043975,
"train_accuracy_std": 0.0009383599680312533,
"val_loss_mean": 0.0884952815023191,
"val_loss_std": 0.018836561490730732,
"val_accuracy_mean": 0.9752344649335718,
"val_accuracy_std": 0.005009718519486596,
"test_loss_mean": 0.31723329968745473,
"test_loss_std": 0.09017346012735108,
"test_accuracy_mean": 0.8885251938016441,
"test_accuracy_std": 0.029840646977007373
```

## Age (8 classes, cross entropy loss)

### Train on IMDB and WIKI

2021-07-21 14:06:39,596 INFO tune.py:549 -- Total run time: 7756.80 seconds (7756.58 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', False), ('dropout', 0.2), ('num_residuals_per_block', 4), ('num_blocks', 0), ('batch_size', 512), ('lr', 0.011587715829876227), ('weight_decay', 0.054480726081375434), ('gamma', 0.8241502276709554), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 8), ('validation_split', 0.1)])
Best trial final validation loss: 1.0482525458702674
Best trial final validation accuracy: 0.6082360326428123

```
2021-07-21 15:02:12,334 - trainer - INFO -     epoch          : 10
2021-07-21 15:02:12,334 - trainer - INFO -     loss           : 0.9352032564335985
2021-07-21 15:02:12,334 - trainer - INFO -     accuracy       : 0.6545405601887413
2021-07-21 15:02:12,334 - trainer - INFO -     val_loss       : 1.0433464042651348
2021-07-21 15:02:12,334 - trainer - INFO -     val_accuracy   : 0.6113181963960291
2021-07-21 15:02:12,369 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0721_145736/checkpoint-epoch10.pth ...
2021-07-21 15:02:12,406 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on IMDB, WIKI, and Adience

2021-07-21 14:11:30,885 INFO tune.py:549 -- Total run time: 8034.66 seconds (8034.45 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', False), ('dropout', 0.2), ('num_residuals_per_block', 4), ('num_blocks', 0), ('batch_size', 512), ('lr', 0.011587715829876227), ('weight_decay', 0.054480726081375434), ('gamma', 0.8241502276709554), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki_adience'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 8), ('validation_split', 0.1)])
Best trial final validation loss: 1.0232807252465226
Best trial final validation accuracy: 0.615121598844209

```
2021-07-21 15:02:38,319 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0721_145747/checkpoint-epoch10.pth ...
2021-07-21 15:03:07,395 - trainer - INFO -     epoch          : 11
2021-07-21 15:03:07,395 - trainer - INFO -     loss           : 0.8974189443562165
2021-07-21 15:03:07,395 - trainer - INFO -     accuracy       : 0.66867144750342
2021-07-21 15:03:07,395 - trainer - INFO -     val_loss       : 1.0200129478442959
2021-07-21 15:03:07,395 - trainer - INFO -     val_accuracy   : 0.6172819530592936
2021-07-21 15:03:07,437 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0721_145747/checkpoint-epoch11.pth ...
2021-07-21 15:03:07,495 - trainer - INFO - Saving current best: model_best.pth .
```

### Train on Adience from scratch (5 times 5-fold cross validation)

saved/0721_152057_cross-val-results.json

```
"train_loss_mean": 0.04190095121983994,
"train_loss_std": 0.032262749448723003,
"train_accuracy_mean": 0.988930769772272,
"train_accuracy_std": 0.009345581732264688,
"val_loss_mean": 0.31300895695269637,
"val_loss_std": 0.059488147284363725,
"val_accuracy_mean": 0.9224977238570715,
"val_accuracy_std": 0.01181242786189685,
"test_loss_mean": 1.8838238582339832,
"test_loss_std": 0.3044851064838116,
"test_accuracy_mean": 0.5510827476376152,
"test_accuracy_std": 0.03818265664112785
```  
  
### Pre-trained on IMDB and WIKI, fine-tune on Adience (5 times 5-fold cross validation)

saved/0721_153547_cross-val-results.json

```
"train_loss_mean": 0.04339913196417851,
"train_loss_std": 0.030123008445561608,
"train_accuracy_mean": 0.9891745066247694,
"train_accuracy_std": 0.008108085307960462,
"val_loss_mean": 0.2532132251602203,
"val_loss_std": 0.03768542762220299,
"val_accuracy_mean": 0.927418347899599,
"val_accuracy_std": 0.012263066647882265,
"test_loss_mean": 1.5619400548899158,
"test_loss_std": 0.20294750497817765,
"test_accuracy_mean": 0.591538459839801,
"test_accuracy_std": 0.02674479992826164
```

## Age (101 classes, cross entropy loss)

### Train on IMDB and WIKI

2021-07-21 17:30:05,991 INFO tune.py:549 -- Total run time: 7617.83 seconds (7617.58 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', False), ('dropout', 0.2), ('num_residuals_per_block', 4), ('num_blocks', 0), ('batch_size', 512), ('lr', 0.011587715829876227), ('weight_decay', 0.054480726081375434), ('gamma', 0.8241502276709554), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 101), ('validation_split', 0.1)])
Best trial final validation loss: 3.270758928396763
Best trial final validation accuracy: 0.14721908349026994


```
2021-07-22 13:25:26,630 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_102338/checkpoint-epoch15.pth ...
2021-07-22 13:37:14,939 - trainer - INFO -     epoch          : 16
2021-07-22 13:37:14,939 - trainer - INFO -     loss           : 2.9365658889313395
2021-07-22 13:37:14,939 - trainer - INFO -     accuracy       : 0.217368259972018
2021-07-22 13:37:14,939 - trainer - INFO -     accuracy_relaxed: 0.644874166712389
2021-07-22 13:37:14,939 - trainer - INFO -     val_loss       : 3.2329624127119017
2021-07-22 13:37:14,939 - trainer - INFO -     val_accuracy   : 0.17277894006969755
2021-07-22 13:37:14,939 - trainer - INFO -     val_accuracy_relaxed: 0.6010960429455208
2021-07-22 13:37:14,979 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_102338/checkpoint-epoch16.pth ...
2021-07-22 13:37:15,026 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on IMDB, WIKI, and Adience

2021-07-21 17:27:40,209 INFO tune.py:549 -- Total run time: 7450.03 seconds (7449.75 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', False), ('dropout', 0.2), ('num_residuals_per_block', 4), ('num_blocks', 0), ('batch_size', 512), ('lr', 0.011587715829876227), ('weight_decay', 0.054480726081375434), ('gamma', 0.8241502276709554), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki_adience'), ('num_samples', 100), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 101), ('validation_split', 0.1)])
Best trial final validation loss: 3.1681374805729563
Best trial final validation accuracy: 0.17755839152419936


```
2021-07-22 15:05:14,299 - trainer - INFO -     epoch          : 22
2021-07-22 15:05:14,300 - trainer - INFO -     loss           : 2.7670100048367856
2021-07-22 15:05:14,300 - trainer - INFO -     accuracy       : 0.2636291253419973
2021-07-22 15:05:14,300 - trainer - INFO -     accuracy_relaxed: 0.6608936816005472
2021-07-22 15:05:14,300 - trainer - INFO -     val_loss       : 3.12917639278784
2021-07-22 15:05:14,300 - trainer - INFO -     val_accuracy   : 0.20624277228763666
2021-07-22 15:05:14,300 - trainer - INFO -     val_accuracy_relaxed: 0.6122513009882254
2021-07-22 15:05:14,348 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_102321/checkpoint-epoch22.pth ...
2021-07-22 15:05:14,403 - trainer - INFO - Saving current best: model_best.pth ...
```


### Train on Adience from scratch (5 times 5-fold cross validation)

saved/0722_164942_cross-val-results.json

```
"train_loss_mean": 0.038432069756377835,
"train_loss_std": 0.027884319821581103,
"train_accuracy_mean": 0.9899485142903532,
"train_accuracy_std": 0.007735580455725201,
"train_accuracy_relaxed_mean": 0.9899485142903532,
"train_accuracy_relaxed_std": 0.007735580455725201,
"val_loss_mean": 0.3050184017853968,
"val_loss_std": 0.0497976109543561,
"val_accuracy_mean": 0.9234693524167783,
"val_accuracy_std": 0.012465248668803556,
"val_accuracy_relaxed_mean": 0.9234693524167783,
"val_accuracy_relaxed_std": 0.012465248668803556,
"test_loss_mean": 1.9645439335921062,
"test_loss_std": 0.31958286856620943,
"test_accuracy_mean": 0.5409089567134056,
"test_accuracy_std": 0.03856700464423375,
"test_accuracy_relaxed_mean": 0.5409089567134056,
"test_accuracy_relaxed_std": 0.03856700464423375
```

### Pre-trained on IMDB and WIKI, fine-tune on Adience (5 times 5-fold cross validation)

saved/0722_171635_cross-val-results.json

```
"train_loss_mean": 0.04338961815139873,
"train_loss_std": 0.03018370021068777,
"train_accuracy_mean": 0.988935596807582,
"train_accuracy_std": 0.007997939199271012,
"train_accuracy_relaxed_mean": 0.988935596807582,
"train_accuracy_relaxed_std": 0.007997939199271012,
"val_loss_mean": 0.24978204337834944,
"val_loss_std": 0.031318021988937884,
"val_accuracy_mean": 0.9283721701306149,
"val_accuracy_std": 0.010990985978302299,
"val_accuracy_relaxed_mean": 0.9283721701306149,
"val_accuracy_relaxed_std": 0.010990985978302299,
"test_loss_mean": 1.5450158784519987,
"test_loss_std": 0.2202436955315273,
"test_accuracy_mean": 0.5884687953460334,
"test_accuracy_std": 0.02461252593349489,
"test_accuracy_relaxed_mean": 0.5884687953460334,
"test_accuracy_relaxed_std": 0.02461252593349489
```


# Evaluation results (pure MLP)

Validation split is always 10%

## Gender (2 classes, cross entropy loss)

### Train on IMDB and WIKI

2021-07-22 21:02:23,368 INFO tune.py:549 -- Total run time: 2597.27 seconds (2597.08 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'gender'), ('only_MLP', True), ('dropout', 0), ('num_residuals_per_block', 1), ('num_blocks', 4), ('batch_size', 512), ('lr', 0.001468989807764881), ('weight_decay', 0.02276685024868625), ('gamma', 0.06637926838138382), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 50), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 2), ('validation_split', 0.1)])
Best trial final validation loss: 0.4447786082059909
Best trial final validation accuracy: 0.8229001883239171

```
2021-07-22 21:51:08,481 - trainer - INFO -     epoch          : 14
2021-07-22 21:51:08,481 - trainer - INFO -     loss           : 0.4391215724628764
2021-07-22 21:51:08,482 - trainer - INFO -     accuracy       : 0.8302272937698891
2021-07-22 21:51:08,482 - trainer - INFO -     accuracy_relaxed: 1.0
2021-07-22 21:51:08,482 - trainer - INFO -     val_loss       : 0.4545662712592345
2021-07-22 21:51:08,482 - trainer - INFO -     val_accuracy   : 0.8224853656124114
2021-07-22 21:51:08,482 - trainer - INFO -     val_accuracy_relaxed: 1.0
2021-07-22 21:51:08,497 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_211202/checkpoint-epoch14.pth ...
2021-07-22 21:51:08,516 - trainer - INFO - Saving current best: model_best.pth ...
```
### Train on IMDB, WIKI, and Adience

2021-07-22 21:08:33,110 INFO tune.py:549 -- Total run time: 2948.80 seconds (2948.23 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'gender'), ('only_MLP', True), ('dropout', 0), ('num_residuals_per_block', 2), ('num_blocks', 2), ('batch_size', 256), ('lr', 0.0010710256549997944), ('weight_decay', 0.04280597490713533), ('gamma', 0.008062359381277659), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki_adience'), ('num_samples', 50), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 2), ('validation_split', 0.1)])
Best trial final validation loss: 0.430916253225935
Best trial final validation accuracy: 0.8319768841801107

```
2021-07-22 21:49:58,929 - trainer - INFO -     epoch          : 11
2021-07-22 21:49:58,930 - trainer - INFO -     loss           : 0.4194352995402841
2021-07-22 21:49:58,930 - trainer - INFO -     accuracy       : 0.8347049324093087
2021-07-22 21:49:58,930 - trainer - INFO -     accuracy_relaxed: 1.0
2021-07-22 21:49:58,930 - trainer - INFO -     val_loss       : 0.42992067410170665
2021-07-22 21:49:58,930 - trainer - INFO -     val_accuracy   : 0.8324708787285806
2021-07-22 21:49:58,930 - trainer - INFO -     val_accuracy_relaxed: 1.0
2021-07-22 21:49:58,949 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_211330/checkpoint-epoch11.pth ...
2021-07-22 21:49:58,974 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on Adience from scratch (5 times 5-fold cross validation)

saved/0722_222743_cross-val-results.json
```
"train_loss_mean": 0.5825609715251807,
"train_loss_std": 0.11320552604289096,
"train_accuracy_mean": 0.5921929309266978,
"train_accuracy_std": 0.13388802517303114,
"val_loss_mean": 0.5829746401828896,
"val_loss_std": 0.11280249339977262,
"val_accuracy_mean": 0.5941951483370282,
"val_accuracy_std": 0.13307684414193133,
"test_loss_mean": 0.6253205517655913,
"test_loss_std": 0.07896039238073754,
"test_accuracy_mean": 0.578727050339925,
"test_accuracy_std": 0.1041487423967318
```

### Pre-trained on IMDB and WIKI, fine-tune on Adience (5 times 5-fold cross validation)

saved/0722_221435_cross-val-results.json
```
"train_loss_mean": 0.07475495609180566,
"train_loss_std": 0.0034093249003471515,
"train_accuracy_mean": 0.9812534086692195,
"train_accuracy_std": 0.001073477600994517,
"val_loss_mean": 0.11475725782585611,
"val_loss_std": 0.016295059004781863,
"val_accuracy_mean": 0.9662740099565249,
"val_accuracy_std": 0.005094744287542971,
"test_loss_mean": 0.2992695421451408,
"test_loss_std": 0.060793215291536876,
"test_accuracy_mean": 0.8921529804166417,
"test_accuracy_std": 0.019196948503006118
```

## Age (8 classes, cross entropy loss)

### Train on IMDB and WIKI

2021-07-22 23:06:12,847 INFO tune.py:549 -- Total run time: 2868.76 seconds (2868.57 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', True), ('dropout', 0), ('num_residuals_per_block', 1), ('num_blocks', 2), ('batch_size', 256), ('lr', 0.0035753161317240803), ('weight_decay', 3.584710633004188e-06), ('gamma', 0.0004325327646962347), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 50), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 8), ('validation_split', 0.1)])
Best trial final validation loss: 1.1113964388003716
Best trial final validation accuracy: 0.5717765222849969

```
2021-07-22 23:09:35,036 - trainer - INFO -     epoch          : 4
2021-07-22 23:09:35,036 - trainer - INFO -     loss           : 1.0589350107990785
2021-07-22 23:09:35,036 - trainer - INFO -     accuracy       : 0.6023642110031296
2021-07-22 23:09:35,036 - trainer - INFO -     val_loss       : 1.1045904438465068
2021-07-22 23:09:35,036 - trainer - INFO -     val_accuracy   : 0.5755550259725906
2021-07-22 23:09:35,051 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_230719/checkpoint-epoch4.pth ...
2021-07-22 23:09:35,065 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on IMDB, WIKI, and Adience

2021-07-22 23:02:41,543 INFO tune.py:549 -- Total run time: 2641.99 seconds (2641.80 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', True), ('dropout', 0), ('num_residuals_per_block', 1), ('num_blocks', 2), ('batch_size', 256), ('lr', 0.0035753161317240803), ('weight_decay', 3.584710633004188e-06), ('gamma', 0.0004325327646962347), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki_adience'), ('num_samples', 50), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 8), ('validation_split', 0.1)])
Best trial final validation loss: 1.099634029382577
Best trial final validation accuracy: 0.576113652781122

```
2021-07-22 23:08:05,917 - trainer - INFO -     epoch          : 4
2021-07-22 23:08:05,917 - trainer - INFO -     loss           : 1.0491730616077994
2021-07-22 23:08:05,917 - trainer - INFO -     accuracy       : 0.6049741187542779
2021-07-22 23:08:05,917 - trainer - INFO -     val_loss       : 1.1036202154276562
2021-07-22 23:08:05,917 - trainer - INFO -     val_accuracy   : 0.5747914242648614
2021-07-22 23:08:05,929 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0722_230556/checkpoint-epoch4.pth ...
2021-07-22 23:08:05,943 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on Adience from scratch (5 times 5-fold cross validation)

saved/0722_231921_cross-val-results.json

```
"train_loss_mean": 1.1818777134307918,
"train_loss_std": 0.040738286630021485,
"train_accuracy_mean": 0.42667301288076714,
"train_accuracy_std": 0.019601423041593215,
"val_loss_mean": 1.212809873234306,
"val_loss_std": 0.029987833890638606,
"val_accuracy_mean": 0.4232847308144483,
"val_accuracy_std": 0.015104964140723194,
"test_loss_mean": 1.4653894133352239,
"test_loss_std": 0.1532473417629615,
"test_accuracy_mean": 0.3916536457639523,
"test_accuracy_std": 0.08538745754507433
```

### Pre-trained on IMDB and WIKI, fine-tune on Adience (5 times 5-fold cross validation)

saved/0722_232910_cross-val-results.json

```
"train_loss_mean": 0.3137280245500197,
"train_loss_std": 0.019206143581372112,
"train_accuracy_mean": 0.900754738082648,
"train_accuracy_std": 0.011260066834356295,
"val_loss_mean": 0.44068487366180586,
"val_loss_std": 0.014698732238824862,
"val_accuracy_mean": 0.8494826128204466,
"val_accuracy_std": 0.012637675231786879,
"test_loss_mean": 1.0887040289923942,
"test_loss_std": 0.04963626723408415,
"test_accuracy_mean": 0.5980103764445449,
"test_accuracy_std": 0.020127804811253428
```

## Age (101 classes, cross entropy loss)

### Train on IMDB and WIKI


2021-07-23 00:05:18,748 INFO tune.py:549 -- Total run time: 2908.05 seconds (2907.85 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', True), ('dropout', 0), ('num_residuals_per_block', 1), ('num_blocks', 2), ('batch_size', 256), ('lr', 0.0035753161317240803), ('weight_decay', 3.584710633004188e-06), ('gamma', 0.0004325327646962347), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki'), ('num_samples', 50), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 101), ('validation_split', 0.1)])
Best trial final validation loss: 3.494242325807229
Best trial final validation accuracy: 0.07319522912743252

```
2021-07-23 00:14:53,570 - trainer - INFO -     epoch          : 4
2021-07-23 00:14:53,570 - trainer - INFO -     loss           : 3.434894827414546
2021-07-23 00:14:53,570 - trainer - INFO -     accuracy       : 0.07994525716521166
2021-07-23 00:14:53,570 - trainer - INFO -     val_loss       : 3.494174796801347
2021-07-23 00:14:53,571 - trainer - INFO -     val_accuracy   : 0.07479777989610964
2021-07-23 00:14:53,584 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0723_001214/checkpoint-epoch4.pth ...
2021-07-23 00:14:53,602 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on IMDB, WIKI, and Adience

2021-07-23 00:06:09,607 INFO tune.py:549 -- Total run time: 2768.49 seconds (2768.31 seconds for the tuning loop).
Best trial config: OrderedDict([('last_activation', None), ('min_bound', None), ('max_bound', None), ('criterion', 'cse'), ('gender_or_age', 'age'), ('only_MLP', True), ('dropout', 0), ('num_residuals_per_block', 1), ('num_blocks', 2), ('batch_size', 256), ('lr', 0.0035753161317240803), ('weight_decay', 3.584710633004188e-06), ('gamma', 0.0004325327646962347), ('data_dir', '/home/tk/repos/age-gender/data'), ('cpus', 8), ('dataset', 'imdb_wiki_adience'), ('num_samples', 50), ('max_num_epochs', 10), ('gpus_per_trial', 1), ('limit_data', None), ('amp', True), ('num_classes', 101), ('validation_split', 0.1)])
Best trial final validation loss: 3.4445898386598364
Best trial final validation accuracy: 0.0911870936672285

```
2021-07-23 00:14:31,880 - trainer - INFO -     epoch          : 3
2021-07-23 00:14:31,880 - trainer - INFO -     loss           : 3.381837900169576
2021-07-23 00:14:31,880 - trainer - INFO -     accuracy       : 0.0992230278918549
2021-07-23 00:14:31,880 - trainer - INFO -     val_loss       : 3.4402417098086304
2021-07-23 00:14:31,881 - trainer - INFO -     val_accuracy   : 0.09004703696847895
2021-07-23 00:14:31,896 - trainer - INFO - Saving checkpoint: saved/models/ResMLP/0723_001225/checkpoint-epoch3.pth ...
2021-07-23 00:14:31,918 - trainer - INFO - Saving current best: model_best.pth ...
```

### Train on Adience from scratch (5 times 5-fold cross validation)

saved/0723_002526_cross-val-results.json

```
"train_loss_mean": 1.6340719273702822,
"train_loss_std": 0.07070868060920533,
"train_accuracy_mean": 0.3086391164157737,
"train_accuracy_std": 0.021769419883324335,
"val_loss_mean": 1.6634917458777645,
"val_loss_std": 0.05151901331374509,
"val_accuracy_mean": 0.30353138495874155,
"val_accuracy_std": 0.014031408228725653,
"test_loss_mean": 1.8142997765515412,
"test_loss_std": 0.13583505408942959,
"test_accuracy_mean": 0.3007147665017866,
"test_accuracy_std": 0.07380871865494001
```


### Pre-trained on IMDB and WIKI, fine-tune on Adience (5 times 5-fold cross validation)

saved/0723_081829_cross-val-results.json

```
"train_loss_mean": 0.44421417788322914,
"train_loss_std": 0.02372537339643898,
"train_accuracy_mean": 0.8417264881033596,
"train_accuracy_std": 0.01714717947285312,
"val_loss_mean": 0.5503468283981173,
"val_loss_std": 0.032659087619235215,
"val_accuracy_mean": 0.7967200603849287,
"val_accuracy_std": 0.021768412713477465,
"test_loss_mean": 1.1914044722275472,
"test_loss_std": 0.0962698188750242,
"test_accuracy_mean": 0.5648832129769986,
"test_accuracy_std": 0.04092359810627668
```