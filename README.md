### 训练

#### WMT14_en-de

##### 没有constraint的training（普通的LevT）

```bash
python train.py data/wmt14.en-de-bin --save-dir checkpoints/wmt14_ende_distill \
--ddp-backend=legacy_ddp --task translation_lev \
--criterion nat_loss --arch levenshtein_transformer \
--noise random_delete \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 \
--warmup-updates 10000 --warmup-init-lr 1e-07 \
--label-smoothing 0.1 --dropout 0.3 --weight-decay 0.01 \
--decoder-learned-pos --encoder-learned-pos --apply-bert-init \
--log-format simple --log-interval 1 --fixed-validation-seed 7 \
--max-tokens 8192 --save-interval-updates 10000 --max-update 300000 \
--source-lang en --target-lang de
```

##### constraint training

```bash
python train.py data/wmt14.en-de-bin --save-dir checkpoints/wmt14_ende_distill_cst \
--ddp-backend=legacy_ddp --task translation_lev \
--criterion nat_loss --arch levenshtein_transformer \
--noise random_delete_wo_cs \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 \
--warmup-updates 10000 --warmup-init-lr 1e-07 \
--label-smoothing 0.1 --dropout 0.3 --weight-decay 0.01 \
--decoder-learned-pos --encoder-learned-pos --apply-bert-init \
--log-format simple --log-interval 1 --fixed-validation-seed 7 \
--max-tokens 8192 --save-interval-updates 10000 --max-update 300000 \
--source-lang en --target-lang de
```

### 环境

配置如普通的levt，参考https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation

1. 从github上clone fairseq
2. 按照fairseq配置标准环境

或者是进入到fairseq文件，pip install --editable ./

### 数据

放在data目录下，wmt14.en-de-bin中，train.constraint 是 每条training数据对应的constraint word，test.constraint 同理