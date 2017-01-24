#!/bin/bash

# usage: ./prepare.sh corpora/train_02-21.txt corpora/dev_22.txt corpora/dev_24.txt corpora/test_23.txt

# One tree per line.
train=$1

train_base=${train##*/}

train_stripped=wsj/${train_base}.stripped
train_vocab=wsj/${train_base}.vocab.gz
train_traversed=wsj/${train_base}.traversed

semi_train_vocab=semi/${train_base}.vocab.gz

mkdir wsj
mkdir semi
# Remove function tags.
if [[ "$train" == *.gz ]]
then
    zcat $train | python strip_function_tags.py > $train_stripped
else
    cat $train | python strip_function_tags.py > $train_stripped
fi

# Download Charniak parser.
python -mbllipparser.ModelFetcher -i WSJ-PTB3 -d wsj


# Create a vocab file.
python create_vocab.py $train_stripped 9 | gzip > $train_vocab
python traversal.py $train_vocab $train_stripped > $train_traversed
python create_vocab.py $train_stripped 1 | gzip > $semi_train_vocab

python traversal.py $semi_train_vocab $train_stripped > semi/${train_base}.traversed

shift
for dev in $@
do
    dev_base=${dev##*/}
    dev_stripped=wsj/${dev_base}.stripped
    dev_nbest=wsj/${dev_base}.nbest.gz

    if [[ "$dev" == *.gz ]]
    then
        zcat $dev | python strip_function_tags.py > $dev_stripped
    else
        cat $dev | python strip_function_tags.py > $dev_stripped
    fi

    # Generate nbest parses with Charniak parser. On a modern processer, parsing
    # section 24 takes about 5 minutes. 
    cat $dev_stripped | python nbest_parse.py | gzip > $dev_nbest

    # Preprocess train, dev and dev_nbest files.
    python traversal.py $train_vocab $dev_stripped > wsj/${dev_base}.traversed
    python traversal.py $train_vocab $dev_stripped $dev_nbest | gzip > wsj/${dev_base}.nbest.traversed.gz

    python traversal.py $semi_train_vocab $dev_stripped > semi/${dev_base}.traversed
    python traversal.py $semi_train_vocab $dev_stripped $dev_nbest | gzip > semi/${dev_base}.nbest.traversed.gz

    # # Path to millions of trees file. One tree per line.
    # SILVER='SET THIS PATH'
    # python sym2id.py semi/train.gz | gzip > semi/sym2id.gz
    # python integerize.py semi/sym2id.gz $SILVER | gzip > semi/silver.gz

    # # Remove unnecessary data.
    # rm wsj/[xyz].gz
done
