#!/bin/bash
candidate_file=$1
output_file=$2

python rescore.py \
  semi/train_02-21.txt.traversed \
  models/semi/model \
  $candidate_file \
  $output_file
