# Evaluation results

Validation split is always 10%

## Gender (2 classes, cross entropy loss)

### Train on IMDB and WIKI

```
train_loss:     0.4422892515230485
train_accuracy: 0.8247924730961401

val_loss:       0.44423754876240706
val_accuracy:   0.8241741959549072
```
### Pre-trained on IMDB and WIKI, fine-tuned on Adience (5 times 5-fold cross validation)

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

### Train on Adience from scratch (5 times 5-fold cross validation)

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

### Train on IMDB, WIKI, and Adience

```
train_loss:     0.42638325623907836
train_accuracy: 0.831892218514716

val_loss:       0.4287793638150385
val_accuracy:   0.8336972114977788
```

## Age (8 classes, cross entropy loss)

### Train on IMDB and WIKI

```
train_loss:     1.0887723597771908
train_accuracy: 0.5783940968645709

val_loss:       1.1134433766429344
val_accuracy:   0.570466669813993
```

### Pre-trained on IMDB and WIKI, fine-tuned on Adience (5 times 5-fold cross validation)

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

### Train on Adience from scratch (5 times 5-fold cross validation)

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

### Train on IMDB, WIKI, and Adience

```
train_loss:     1.0673519349163514
train_accuracy: 0.5878167265879131

val_loss:       1.0959079291309892
val_accuracy:   0.5741907310982413
```

## Age (101 classes, cross entropy loss)

### Train on IMDB and WIKI

```
train_loss:             3.3405551750547704
train_accuracy:         0.1458298327252277
train_accuracy_relaxed: 0.5593033732716998

val_loss:               3.4325964542535634
val_accuracy:           0.12708819148043352
val_accuracy_relaxed:   0.5443172292625807
```

### Pre-trained on IMDB and WIKI, fine-tuned on Adience (5 times 5-fold cross validation)

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

### Train on Adience from scratch (5 times 5-fold cross validation)

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

### Train on IMDB, WIKI, and Adience

```
train_loss:             3.2645355109150858
train_accuracy:         0.16898405437756497
train_accuracy_relaxed: 0.5740050017099864

val_loss:               3.350942190100507
val_accuracy:           0.1496662111017662
val_accuracy_relaxed:   0.5579260079373424
```