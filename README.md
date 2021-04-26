# Variational Recurrent Topic Model
<!-- ![alt-text-1]("title-1") ![alt-text-2](image2.png "title-2") -->
The implementation of NeurIPS 2020 paper 
["A Discrete Variational Recurrent Topic Model without the Reparametrization Trick".](https://arxiv.org/abs/2010.12055)

This Tensorflow code implements the model and reproduces the results from the paper.

<img src="figs/vrtm_Graph.png" width="265"/> <img src="figs/vrtm_EncDec.png" width="550"/> 
# Citing this work
Please cite if you find our work helpful to your research:

    @article{rezaee2020discrete,
      title={A Discrete Variational Recurrent Topic Model without the Reparametrization Trick},
      author={Rezaee, Mehdi and Ferraro, Francis},
      journal={Advances in neural information processing systems},
      year={2020}
    }
# Requirements
- python3.5 
- tensorflow 1.13
- gensim 3.8.8

# Data Format
- One line per document
- [ACL2017 Paper dataset (AP News, BNC and IMDB)](https://drive.google.com/drive/folders/1n4s1Tz3RcJFmp2Itg5MWjoCMe_KYEbfj)

## main.py arguments:
```
usage: main.py [-h] [--dataset DATASET] [--batch_size BATCH_SIZE]
               [--num_epochs NUM_EPOCHS] [--frequency_limit FREQUENCY_LIMIT]
               [--max_seqlen MAX_SEQLEN] [--num_units NUM_UNITS]
               [--num_hidden NUM_HIDDEN] [--dim_emb DIM_EMB]
               [--num_topics NUM_TOPICS] [--num_layers NUM_LAYERS]
               [--learning_rate LEARNING_RATE] [--dropout DROPOUT]
               [--rnn_model RNN_MODEL] [--decay_epoch DECAY_EPOCH]
               [--lstm_norm LSTM_NORM] [--prior PRIOR]
               [--generate_len GENERATE_LEN] [--init_from INIT_FROM]
               [--save_dir SAVE_DIR] [--use_word2vec]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset apnews,imdb, bnc
  --batch_size BATCH_SIZE
                        batch size
  --num_epochs NUM_EPOCHS
                        number of epochs
  --frequency_limit FREQUENCY_LIMIT
                        word frequency limit for vocabulary
  --max_seqlen MAX_SEQLEN
                        maximum sequence length
  --num_units NUM_UNITS
                        num of units
  --num_hidden NUM_HIDDEN
                        hidden units of inference network
  --dim_emb DIM_EMB     dimension of embedding
  --num_topics NUM_TOPICS
                        number of topics
  --num_layers NUM_LAYERS
                        number of layers
  --learning_rate LEARNING_RATE
                        learning rate
  --dropout DROPOUT     dropout
  --rnn_model RNN_MODEL
                        GRU,LSTM, RNN Cells
  --decay_epoch DECAY_EPOCH
                        adaptive learning rate decay epoch
  --lstm_norm LSTM_NORM
                        Using LayerNormBasicLSTMCell instead of LSTMCell
  --prior PRIOR         prior coefficient
  --generate_len GENERATE_LEN
                        The length of the sentence to generate
  --init_from INIT_FROM
                        init_from
  --save_dir SAVE_DIR   dir for saving the model
  --use_word2vec        use word2vec
```
# Running the code:
To recreate our model results (Table 1, Table 2) from the paper, run the following with the corresponding parameters e.g., GPU 1:

$DATA : IMDB, APNEWS, BNC

$FREQUENCY_LIMIT: 112 (IMDB), 120 (APNEWS), (111) BNC

$DIM_EMB: 300, 400

$NUM_TOPICS: 10, 30, 50

$RNN_MODEL: LSTM, basicRNN, GRU

Use --use_word2vec for word2vec pre-trained word embeddings.
```
CUDA_VISIBLE_DEVICES=1 python main.py --prior 0.5 --dataset $DATA --batch_size 100 --num_epochs 40 --frequency_limit $FREQUENCY_LIMIT --max_seqlen 45 --num_units 600 --num_hidden 500 --dim_emb $DIM_EMB --num_topics $NUM_TOPICS --num_layers 1 --learning_rate 0.001 --dropout 0.5 --RNN_MODEL $RNN_MODEL  --decay_epoch 20 --lstm_norm 0 --generate_len 60
```

