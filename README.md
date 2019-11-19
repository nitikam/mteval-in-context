
#Install requirements: 
pip install -r requirements

#add ctxteval lib to pythonpath
export PYTHONPATH=${PYTHONPATH}:path_to_ctxteval
cd path_to_ctxteval/experiments
```


#To train:

```shell
config_file=att_mt_bert_en.json  #sample config files for the ESIM/ATT model are in the experiments directory

serialization_dir=temp
cuda_device=0

#all data files need to be in the jsonl format with keys "ref", "mt" and "score". Samples in the data directory
#To change the train/valid data paths, change the config file, or add overrides to the train command
train=../data/train.jsonl
valid=../data/valid.jsonl


#the train command creates the model.tar.gz file in the serialization dir (among other files)
allennlp train ${config_file} -s ${serialization_dir} --include-package ctxteval -o "{train_data_path: $train, valid_data_path: $valid, trainer: {cuda_device: $cuda_device}}"
```

#To predict:
```shell
pred=~/data/pred.jsonl
out=pred.seg.scores 
#this creates a file out with scores for each sentence in $pred
allennlp predict ${serialization_dir}/model.tar.gz $pred \
            --output-file $out \
            --cuda-device $cuda_device \
            --predictor mteval-predictor \
            --include-package ctxteval \
            --batch-size 32 
            --silent \

```
 