# ESC4NMT

AAAI 2020: Explicit Sentence Compression for Neural Machine Translation

This implementation is based on [fairseq](https://github.com/pytorch/fairseq). We take en2de NMT experiment for example.

### Requirements

- OS: macOS or Linux
- NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
- Pytorch

### Preparation

1. Download and Preprocess WMT 14 data

    The WMT English to German dataset can be preprocessed using the `prepare-wmt14en2de.sh` script.
    By default it will produce a dataset that was modeled after [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), but with additional news-commentary-v12 data from WMT'17.

    To use only data available in WMT'14 or to replicate results obtained in the original [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) paper, please use the `--icml17` option.

    ```bash
    # Download and prepare the data
    cd examples/translation/
    # WMT'17 data:
    bash prepare-wmt14en2de.sh
    # or to use WMT'14 data:
    # bash prepare-wmt14en2de.sh --icml17

    cd  wmt14_en_de
    # cd wmt17_en_de

    mkdir ./tmp/esc/

    sed -r 's/(@@ )|(@@ ?$)//g' train.en > ./tmp/esc/train.esc.tok.en
    sed -r 's/(@@ )|(@@ ?$)//g' valid.en > ./tmp/esc/valid.esc.tok.en
    sed -r 's/(@@ )|(@@ ?$)//g' test.en > ./tmp/esc/test.esc.tok.en
    ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ./tmp/esc/train.esc.tok.en > ./tmp/esc/train.esc.en
    ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ./tmp/esc/valid.esc.tok.en > ./tmp/esc/valid.esc.en
    ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ./tmp/esc/test.esc.tok.en > ./tmp/esc/test.esc.en

    rm ./tmp/esc/train.esc.tok.en
    rm ./tmp/esc/valid.esc.tok.en
    rm ./tmp/esc/test.esc.tok.en

    ```


2. Perform explicit sentence compression

    ```

    CUDA_VISIBLE_DEVICES=0 python ./scripts/generate_esc.py --esc_model_path ./pretrain/model/esc_giga/ --esc_max_len_a 0.6 --esc_max_len_b 0 --esc_min_len 5 --input_path ./examples/translation/wmt14_en_de/tmp/esc/train.esc.en --output_path ./examples/translation/wmt14_en_de/tmp/esc/train.en-esc

    CUDA_VISIBLE_DEVICES=0 python ./scripts/generate_esc.py --esc_model_path ./pretrain/model/esc_giga/ --esc_max_len_a 0.6 --esc_max_len_b 0 --esc_min_len 5 --input_path ./examples/translation/wmt14_en_de/tmp/esc/valid.esc.en --output_path ./examples/translation/wmt14_en_de/tmp/esc/valid.en-esc


    CUDA_VISIBLE_DEVICES=0 python ./scripts/generate_esc.py --esc_model_path ./pretrain/model/esc_giga/ --esc_max_len_a 0.6 --esc_max_len_b 0 --esc_min_len 5 --input_path ./examples/translation/wmt14_en_de/tmp/esc/test.esc.en --output_path ./examples/translation/wmt14_en_de/tmp/esc/test.en-esc

    BPEROOT=subword-nmt/subword_nmt

    python $BPEROOT/apply_bpe.py -c ./wmt14_en_de/code < ./wmt14_en_de/tmp/esc/train.en-esc > ./wmt14_en_de/tmp/esc/bpe.train.en-esc
    python $BPEROOT/apply_bpe.py -c ./wmt14_en_de/code < ./wmt14_en_de/tmp/esc/valid.en-esc > ./wmt14_en_de/tmp/esc/bpe.valid.en-esc
    python $BPEROOT/apply_bpe.py -c ./wmt14_en_de/code < ./wmt14_en_de/tmp/esc/test.en-esc > ./wmt14_en_de/tmp/esc/bpe.test.en-esc

    cp ./wmt14_en_de/tmp/esc/bpe.train.en-esc ./wmt14_en_de/train.en-esc
    cp ./wmt14_en_de/tmp/esc/bpe.valid.en-esc ./wmt14_en_de/valid.en-esc
    cp ./wmt14_en_de/tmp/esc/bpe.test.en-esc ./wmt14_en_de/test.en-esc

    ```

3. Binarize the dataset

```
TEXT=./examples/translation/wmt14_en_de
python fairseq_cli/multicontext_preprocess.py --source-lang en --target-lang de --source-context en-esc --trainpref $TEXT/train --validpref $TEXT/valid --destdir data-bin/wmt14_en_de_esc --thresholdtgt 0 --thresholdsrc 0 --joined-dictionary --workers 20
```

### Model training

```
EXP_NAME=esc4nmt
EXP_ID=multicontext_transformer_wmt14_en_de
UPDATES=200000
DATA_PATH=data-bin/wmt14_en_de_esc/ 
SAVE_PATH=./checkpoints/$EXP_NAME/${EXP_ID}_up${UPDATES}
LOG_PATH=./logs/$EXP_NAME/${EXP_ID}_up${UPDATES}.log

if [ ! -d ./checkpoints/$EXP_NAME/ ];
then 
    mkdir -p ./checkpoints/$EXP_NAME/
fi

if [ ! -d ./logs/$EXP_NAME/ ];
then 
    mkdir -p ./logs/$EXP_NAME/
fi


CUDA_VISIBLE_DEVICES=`0,1,2,3,4,5,6,7` python train.py ${DATA_PATH} \
                --task multicontext_translation --arch multicontext_transformer_wmt_en_de --share-all-embeddings --dropout 0.1 \
                --source-lang en --target-lang de --source-context en-esc \
                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
                --lr 0.0007 --min-lr 1e-09 \
                --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
                --max-tokens 4096 --save-dir ${SAVE_PATH} \
                --update-freq 1 --no-progress-bar --log-format json --log-interval 50 \
                --save-interval-updates 1000 --keep-interval-updates 200 --max-update ${UPDATES} > $LOG_PATH
```


### Inference

```
python scripts/average_checkpoints.py --inputs ./checkpoints/esc4nmt/multicontext_transformer_wmt14_en_de_up200000/ --num-epoch-checkpoints 5 --output ./checkpoints/esc4nmt/multicontext_transformer_wmt14_en_de_up200000/averaged_model.pt

CKPT=averaged_model.pt

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/multicontext_translate.py data-bin/wmt14_en_de_esc/ --task multicontext_translation --source-lang en --target-lang de --source-context en-esc --path ./checkpoints/esc4nmt/multicontext_transformer_wmt14_en_de_up200000/${CKPT} --buffer-size 2000 --batch-size 128 --beam 5 --remove-bpe --lenpen 0.6 --input ./examples/translation/wmt14_en_de/test.en --other-input ./examples/translation/wmt14_en_de/test.en-esc --output ./result/wmt14_ende_test.de.pred

```

### Reference

```
@inproceedings{li2020explicit,
title={Explicit Sentence Compression for Neural Machine Translation},
author={Zuchao Li and Rui Wang and Kehai Chen and Masao Utiyama and Eiichiro Sumita and Zhuosheng Zhang and Hai Zhao},
booktitle={the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-2020)},
year={2020}
}
```