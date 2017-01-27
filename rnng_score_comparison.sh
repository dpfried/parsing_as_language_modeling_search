#!/bin/bash

echo "wsj 50, 10"
python rnng_score_comparison.py expts/beam_size=50-at_word=5.stderr_all expts/beam_size=50-at_word=5.decode_all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.likelihoods
echo

echo "wsj 100, 10"
python rnng_score_comparison.py expts/beam_size=100-at_word=10.stderr_block--all expts/beam_size=100-at_word=10.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.likelihoods
echo

echo "wsj 200, 20"
python rnng_score_comparison.py expts/beam_size=200-at_word=20.stderr_block--all expts/beam_size=200-at_word=20.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.likelihoods
echo

echo "wsj 400, 40"
python rnng_score_comparison.py expts/beam_size=400-at_word=40.stderr_block--all expts/beam_size=400-at_word=40.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.likelihoods
echo
echo

echo "semi 50, 5"
python rnng_score_comparison.py expts/semi_beam_size=50-at_word=5.stderr_block--all expts/semi_beam_size=50-at_word=5.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.semi.likelihoods
echo

echo "semi 100, 10"
python rnng_score_comparison.py expts/semi_beam_size=100-at_word=10.stderr_block--all expts/semi_beam_size=100-at_word=10.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.semi.likelihoods
echo

echo "semi 200, 20"
python rnng_score_comparison.py expts/semi_beam_size=200-at_word=20.stderr_block--all expts/semi_beam_size=200-at_word=20.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.semi.likelihoods
echo

echo "semi 400, 40"
python rnng_score_comparison.py expts/semi_beam_size=400-at_word=40.stderr_block--all expts/semi_beam_size=400-at_word=40.decode_block--all wsj/dev_22.txt.stripped dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.semi.likelihoods
