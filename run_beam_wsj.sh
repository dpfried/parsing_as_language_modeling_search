#!/bin/bash

beam_size=$1
at_word=$2

block_count=200

block_num=$3
prefix=expts/beam_size=${beam_size}-at_word=${at_word}

python -u search.py \
    --beam_size $beam_size \
    --beam_within_word \
    --beam_size_at_word $at_word \
    --block_count $block_count \
    --decode_file ${prefix}.decode \
    --beam_output_file ${prefix}.beam \
    --block_num ${block_num} \
    > ${prefix}.stdout_block-${block_num} \
    2> ${prefix}.stderr_block-${block_num}
