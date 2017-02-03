#!/bin/bash
candidate_file=$1
output_file=$2

python rescore.py \
  wsj/train_02-21.txt.traversed \
  models/wsj/model \
  $candidate_file \
  $output_file
