# summer2018_nmt
Modified NMT seq2seq model used in summer 2018 research project with the ECE Department, UCLA

This code is derived from the Tensorflow tutorial on NMT seq2seq model. The original codebase can be found [here](https://github.com/tensorflow/nmt). We recommend reading through the above tutorial to understand the basic concepts in the seq2seq model before using this module. 

To adapt the NMT seq2seq model to our problem, we implemented a multiple encoder architecture and added some new evaluation metrics. 

## Requirements
To use the modules in this repository, make sure you have the following packages installed. We recommend using Anaconda as the package manager. 

* [Python 3.5 or above](https://www.python.org/)
* [Tensorflow 1.4 or above](https://www.tensorflow.org/)
* [python_Levenshtein](https://pypi.org/project/python-Levenshtein/)

## What is in this repository
This repository contains the Python modules used to train seq2seq encoder-decoder models and infer with trained models. The **data** subdirectory contains a dataset used to train and test a 2-encoder network for sequences of block length 20. The **models** subdirectory contains models trained on the above data. The models above are able to outperform SCS and SMAP on reconstructing block length 20 sequences from 2 traces in terms of edit distance, for deletion channel probability from 0.1 to 0.5. 


## How to use
To use this module, first clone or download this repository onto your machine. Once downloaded, navigate into the repository untill you see the subdirectory named **nmt**. Then, display the help message for this module by calling: 
``` shell
python nmt.nmt -h 
```

### Example 1
The following command trains a seq2seq model that take an input trace and reconstruct a sequence. The network uses unidirectional encoders, 2 layers of LSTM cells for both the encoder and decoder, 128 units for each LSTM layers, 0.2 dropout rate during training, Adam optimizer with learning rate 0.0001, and luong multiplicative attention. The network will train for 40000 gradient step, outputing basic statistics like training speed, perplexity, etc, every 100 steps, and performs an external evaluation on the evaluation datset using the edit distance metric every 500 steps. 

``` shell
python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./path/to/vocab/vocab --share_vocab=True --train_prefix=./path/to/training/data/train --dev_prefix=./path/to/evaluation/data/val --test_prefix=./path/to/testing/data/test --out_dir=./path/to/where/you/want/to/store/model --num_train_steps=40000 --steps_per_stats=100 --steps_per_external_eval=500 --num_layers=2 --num_units=128 --dropout=0.2 --encoder_type=uni --optimizer=adam --learning_rate=0.0001 --src_max_len=30 --tgt_max_len=30 --src_max_len_infer=30 --num_traces=1 --metrics=edit_distance
```

The training trace data file is ```./path/to/training/data/train.trace```

The training label data file is ```./path/to/training/data/train.label```

All other data files are specified in a similar manner.


### Exmaple 2
The following command trains a seq2seq model that takes 2 input traces and attempts to reconstruct a sequence. This set of arguments will use the multiple encoder architecture, with the encoder states averaged before passing to the decoder. 

``` shell
python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/data/binary_seq_vocab/vocab --share_vocab=True --train_prefix=./nmt/data/data20_2t_large/data0.1/full_seq/train --dev_prefix=./nmt/data/data20_2t_large/data0.1/full_seq/val --test_prefix=./nmt/data/data20_2t_large/data0.1/full_seq/test --out_dir=./nmt/models/model20_2encoder_avg_large/model0.1 --num_train_steps=40000 --steps_per_stats=100 --steps_per_external_eval=500 --num_layers=2 --num_units=128 --dropout=0.2 --encoder_type=uni --optimizer=adam --learning_rate=0.0001 --src_max_len=30 --tgt_max_len=30 --src_max_len_infer=30 --num_traces=2 --NEncoderMode=avg --metrics=edit_distance
```

The training trace data files are ```./nmt/data/data20_2t_large/data0.1/full_seq/train.trace0``` and ```./nmt/data/data20_2t_large/data0.1/train.trace1```

The training label data files are ```./nmt/data/data20_2t_large/data0.1/full_seq/train.label0``` and ```./nmt/data/data20_2t_large/data0.1/train.label1```

**Note:** This repository includes the necessary datasets to directly run the Example 2 training command. Please refer to the ```nmt/data``` subdirectory to see how data files should be foratted and structure. Scripts used in the data generation pipeline can be found [here](https://github.com/LeonAlexandre/seq-data-scripts).

### Example 3
The following command uses a trained seq2seq 2-encoder model specified by ```outdir``` to do inference on new input data. After inference, the output data is evaluated against the reference data using metrics edit distance and hamming distance. 

``` shell
python -m nmt.nmt --out_dir=./nmt/models/model20_2encoder_avg_large/model0.1/best_edit_distance/ --inference_input_file=./nmt/data/data20_2t_large/data0.1/full_seq/test --inference_output_file=./data20_2t_large_infer/test --inference_ref_file=./nmt/data/data20_2t_large/data0.1/full_seq/test.label --metrics=edit_distance,hamming_distance
```

**Note:** This repository include the necessary trained model to directly run the Example 3 inference command. 

## Miscellaneous

The trained models are stored as trained model checkpoints. Because the checkpoint data includes the weights and parameters of the network, the disk scpace usage might become very large. Consider limiting the number of checkpoints to save using ```--num_keep_ckpts```.

The memory usage for this module is quite small when the sequence block lengths are relatively small (< 300). During the training of the models included in this repository, the memory usage did not exceed 1.5Gb.
