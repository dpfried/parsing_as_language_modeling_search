#!/bin/bash

beam_size=50
at_word=5

block_count=100

block_num=$1
prefix=expts/beam_size=${beam_size}-at_word=${at_word}

python -u search.py \
    --beam_size $beam_size \
    --beam_within_word \
    --beam_size_at_word $at_word \
    --block_count $block_count \
    --decode_file ${prefix}.decode \
    --block_num $1 \
    > ${prefix}.stdout_block-${block_num} \
    2> ${prefix}.stderr_block-${block_num}
