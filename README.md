# Classifying Relations by Ranking with Convolutional Neural Networks

Implementation of ACL 2015 Paper:  
[Classifying Relations by Ranking with Convolutional Neural Networks](https://arxiv.org/abs/1504.06580)

## Download SemEval 2010 Task 8 Dataset for Relation Classification
Here is the link to download this dataset:
[link](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50) 

You will also need to download some pre-trained embeddings like 
[GloVe](https://nlp.stanford.edu/projects/glove/). 

## Dependencies 
```
tensorflow (1.3.0)
spacy
pandas
numpy
scikit-learn
```

## Training
Update paths in `model_config.yml`, then start training as: 

```
python3 -m train_crcnn
```


## Evaluation
Once the training is finished and you have trained models in your model directory,
evaluate a model as: 

```
python3 -m test_crcnn --config_file <full path of saved .yml config file in your model directory --model_name <checkpoint prefix of the model you want to evaluate>

```
