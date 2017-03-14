#!/bin/bash

basepath=$1
basename=${basepath##*/}

outpath=$2

max_block_num=$3

output_file=${outpath}/${basename}-all

# decode_file=expts/${basename}.decode_all
# stderr_file=expts/${basename}.stderr_all

for block_num in `seq 0 $max_block_num`; do cat ${basepath}${block_num}; done > $output_file

wc -l $output_file
