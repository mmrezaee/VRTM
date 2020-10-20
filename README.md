# Variational Recurrent Topic Model
<!-- ![alt-text-1]("title-1") ![alt-text-2](image2.png "title-2") -->
<img src="figs/vrtm_Graph.png" width="265"/> <img src="figs/vrtm_EncDec.png" width="550"/> 
# Requirements
- python3.5 
- tensorflow 1.13

# Data Format
- One line per document
- [ACL2017 Paper dataset (AP News, BNC and IMDB)](https://ibm.box.com/s/ls61p8ovc1y87w45oa02zink2zl7l6z4)

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
               [--save_dir SAVE_DIR]

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
```
# Running the code:
```
CUDA_VISIBLE_DEVICES=1 python main.py --prior 0.5 --dataset apnews --batch_size 100 --num_epochs 40 --frequency_limit 120 --max_seqlen 45 --num_units 600 --num_hidden 500 --dim_emb 400 --num_topics 50 --num_layers 1 --learning_rate 0.001 --dropout 0.5 --rnn_model LSTM  --decay_epoch 20 --lstm_norm 0 --generate_len 60
```

