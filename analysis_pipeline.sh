#!/bin/bash

beam_size=$1
at_word=$2

# savio scp -Cr "tomato:~mitchell/backup/choe" ~/projects/rnng_jobs/backup

./concat.sh ../rnng_jobs/backup/choe/silver/stderr/semi_beam_size=${beam_size}-at_word=${at_word}.stderr_block-
./concat.sh ../rnng_jobs/backup/choe/silver/decode/semi_beam_size=${beam_size}-at_word=${at_word}.decode_block-
python decode_analysis.py expts/semi_beam_size=${beam_size}-at_word=${at_word}.stderr_block--all expts/semi_beam_size=${beam_size}-at_word=${at_word}.decode_block--all wsj/dev_22.txt.stripped
