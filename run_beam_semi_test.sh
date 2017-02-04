#!/bin/bash

beam_size=$1
at_word=$2

block_count=2416

block_num=$3
prefix=expts/test_semi_beam_size=${beam_size}-at_word=${at_word}

python -u search.py \
    --beam_size $beam_size \
    --beam_within_word \
    --beam_size_at_word $at_word \
    --block_count $block_count \
    --decode_file ${prefix}.decode \
    --beam_output_file ${prefix}.beam \
    --block_num ${block_num} \
    --data_path semi \
    --model_path models/semi/model \
    --train_path semi/train_02-21.txt.traversed.gz \
    --valid_path semi/test_23.txt.traversed.gz \
    --valid_nbest_path semi/test_23.txt.nbest.gz \
    --valid_nbest_traversed_path semi/test_23.txt.nbest.traversed.gz \
    --gold_dev_stripped_path wsj/test_23.txt.stripped \
    > ${prefix}.stdout_block-${block_num} \
    2> ${prefix}.stderr_block-${block_num}
