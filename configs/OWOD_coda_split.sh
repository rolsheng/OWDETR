#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWDETR_t1
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir ${EXP_DIR}  --eval_every 1 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 4 --train_set 't1_train' --test_set 'val'  \
    --unmatched_boxes --epochs 30 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'ckpt/pretrained_weight.pth' \
    ${PY_ARGS} 

EXP_DIR=exps/OWDETR_t2
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR}  --eval_every 1 \
    --PREV_INTRODUCED_CLS 4 --CUR_INTRODUCED_CLS 11 --train_set 't2_train' --test_set 'val'  \
    --unmatched_boxes --epochs 60 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'exps/OWDETR_t1/checkpoint0029.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t2_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --eval_every 1 \
    --PREV_INTRODUCED_CLS 4 --CUR_INTRODUCED_CLS 11 --train_set 't2_ft' --test_set 'val' \
    --unmatched_boxes --epochs 80 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'exps/OWDETR_t2/checkpoint0059.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t3
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR}  --eval_every 1 \
    --PREV_INTRODUCED_CLS 15 --CUR_INTRODUCED_CLS 7  --train_set 't3_train' --test_set 'val' \
    --unmatched_boxes --epochs 110 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'exps/OWDETR_t2_ft/checkpoint0079.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t3_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --eval_every 1 \
    --PREV_INTRODUCED_CLS 15 --CUR_INTRODUCED_CLS 7  --train_set 't3_ft' --test_set 'val' \
    --unmatched_boxes --epochs 130 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'exps/OWDETR_t3/checkpoint0109.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t4
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --eval_every 1 \
    --PREV_INTRODUCED_CLS 22 --CUR_INTRODUCED_CLS 8 --train_set 't4_train' --test_set 'val' \
    --unmatched_boxes --epochs 160 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'exps/OWDETR_t3_ft/checkpoint0129.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWDETR_t4_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --eval_every 1 \
    --PREV_INTRODUCED_CLS 22 --CUR_INTRODUCED_CLS 8  --train_set 't4_ft' --test_set 'val' \
    --unmatched_boxes --epochs 180 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --pretrain 'exps/OWDETR_t4/checkpoint0159.pth' \
    ${PY_ARGS}
