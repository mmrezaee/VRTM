#!/bin/bash
CUDA_VISIBLE_DEVICES=1\
    python main.py\
    --prior 0.5\
    --dataset apnews\
    --batch_size 100\
    --num_epochs 2\
    --frequency_limit 120\
    --max_seqlen 45\
    --num_units 600\
    --num_hidden 500\
    --dim_emb 400\
    --num_topics 5\
    --num_layers 1\
    --learning_rate 0.001\
    --dropout 0.5\
    --rnn_model LSTM \
    --decay_epoch 20\
    --lstm_norm 0\
    --generate_len 60\
