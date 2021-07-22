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
python train.py /home/zc/projects/distill-data/wmt14-en-de-distill-bin --save-dir checkpoints/wmt14_ende_distill_cst \
--task translation_lev --criterion nat_loss --arch levenshtein_transformer \
--noise random_delete_wo_cs --optimizer adam --adam-betas '(0.9,0.98)' \
--lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 10000 --warmup-init-lr 1e-07 \
--label-smoothing 0.1 --dropout 0.3 --weight-decay 0.01 \
--decoder-learned-pos --encoder-learned-pos --apply-bert-init \
--log-format simple --log-interval 1 --fixed-validation-seed 7 \
--max-tokens 8192 --save-interval-updates 10000 --max-update 300000 \
--source-lang en --target-lang de 
```

#### WMT17_en-zh

##### 没有constraint的training（普通的LevT）

```bash
python train.py data/wmt17.en-zh-bin --save-dir checkpoints/wmt17_enzh_distill \
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
--source-lang en --target-lang zh
```

##### constraint training

```bash
python train.py data/wmt17.en-zh-bin --save-dir checkpoints/wmt17_enzh_distill_cst \
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
--source-lang en --target-lang zh
```

### 生成

#### WMT14 ende

##### 普通 levt 

- no constraints (--mode=0)

```
python generate.py data/wmt14-en-de-distill-bin \
--gen-subset test --task translation_lev \
--path checkpoints/wmt14_ende_distill/checkpoint_best.pt \
--iter-decode-max-iter 9 --iter-decode-eos-penalty 0 \
--beam 1 --remove-bpe --print-step --batch-size 400 \
--cpu --decoding-mode 0
```

- soft constraints (--mode=1)
- hard constraints (--mode=2)

##### cst levt

- no constraints

```
python generate.py data/wmt14-en-de-distill-bin \
--gen-subset test --task translation_lev \
--path checkpoints/wmt14_ende_distill_cst/checkpoint_best.pt \
--iter-decode-max-iter 9 --iter-decode-eos-penalty 0 \
--beam 1 --remove-bpe --print-step --batch-size 400 \
--cpu --decoding-mode 0
```

- soft constraints (--mode=1)
- hard constraints (--mode=2)

#### WMT17 enzh

##### 普通levt 

- no constraints

  ```
  python generate.py /home/zc/projects/nlg/nmt/constrained-levt/distill-data/wmt17.en-zh-bin \
  --gen-subset test --task translation_lev \
  --path checkpoints/wmt17_enzh_distill/checkpoint_best.pt \
  --iter-decode-max-iter 9 --iter-decode-eos-penalty 0 \
  --beam 1 --remove-bpe --print-step --batch-size 400 \
  --cpu --decoding-mode 0
  ```

- soft constraints (--mode=1)
- hard constraints (--mode=2)

##### cst levt

- no constraints

```
python generate.py /home/zc/projects/nlg/nmt/constrained-levt/distill-data/wmt17.en-zh-bin \
--gen-subset test --task translation_lev \
--path checkpoints/wmt17_enzh_distill_cst/checkpoint_best.pt \
--iter-decode-max-iter 9 --iter-decode-eos-penalty 0 \
--beam 1 --remove-bpe --print-step --batch-size 400 \
--cpu --decoding-mode 0
```

- soft constraints (--mode=1)
- hard constraints (--mode=2)

### 环境

配置如普通的levt，参考https://github.com/pytorch/fairseq/tree/master/examples/nonautoregressive_translation

1. 从github上clone fairseq
2. 按照fairseq配置标准环境

或者是进入到fairseq文件，pip install --editable ./

### 数据

放在data目录下，wmt14.en-de-bin中，train.constraint 是 每条training数据对应的constraint word，test.constraint 同理

- 下载：https://cip20w1dt9.feishu.cn/space/folder/fldcneDkbfiUv07F3p1T3nTs18f?from=auth_notice

