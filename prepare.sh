#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "echo ./prepare.sh wsj-train.gz wsj-dev.gz"
    exit
fi

# One tree per line.
TRAIN=$1
DEV=$2

train_stripped=wsj/${TRAIN}.stripped.gz
train_vocab=wsj/${TRAIN}.vocab.gz
dev_stripped=wsj/${DEV}.stripped.gz
dev_nbest=wsj/${DEV}.nbest.gz

train_traversed=wsj/${TRAIN}.traversed.gz
dev_traversed=wsj/${DEV}.traversed.gz
dev_nbest_traversed=wsj/${DEV}.nbest.traversed.gz

mkdir wsj
# Remove function tags.
if [[ "$TRAIN" == *.gz ]]
then
   zcat $TRAIN | python strip_function_tags.py | gzip > wsj/x.gz
   zcat $DEV | python strip_function_tags.py | gzip > wsj/y.gz
else
   cat $TRAIN | python strip_function_tags.py | gzip > $train_stripped
   cat $DEV | python strip_function_tags.py | gzip > $dev_stripped
fi

# Download Charniak parser.
python -mbllipparser.ModelFetcher -i WSJ-PTB3 -d wsj

# Generate nbest parses with Charniak parser. On a modern processer, parsing
# section 24 takes about 5 minutes. 
zcat $dev_stripped | python nbest_parse.py | gzip > $dev_nbest

# Create a vocab file.
python create_vocab.py $train_stripped 9 | gzip > $train_vocab

# Preprocess train, dev and dev_nbest files.
python traversal.py $train_vocab $train_stripped | gzip > $train_traversed
python traversal.py $train_vocab $dev_stripped | gzip > $dev_traversed
python traversal.py $train_vocab $dev_stripped $dev_nbest | gzip > $dev_nbest_traversed
if false; then
    mkdir semi
    python create_vocab.py $train_stripped 1 | gzip > semi/vocab.gz

    python traversal.py semi/vocab.gz $train_stripped | gzip > semi/${TRAIN}.traversed.gz
    python traversal.py semi/vocab.gz $dev_stripped | gzip > semi/${DEV}.traversed.gz
    python traversal.py semi/vocab.gz $dev_stripped $dev_nbest | \
	gzip > semi/${DEV}.nbest.traversed.gz

    # # Path to millions of trees file. One tree per line.
    # SILVER='SET THIS PATH'
    # python sym2id.py semi/train.gz | gzip > semi/sym2id.gz
    # python integerize.py semi/sym2id.gz $SILVER | gzip > semi/silver.gz
fi

# # Remove unnecessary data.
# rm wsj/[xyz].gz
