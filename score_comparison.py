
# coding: utf-8

# In[1]:

# section 22, beam decode from RNNG discriminative model
rnng_gold_file = '../rnng/corpora/22.auto.clean'
rnng_samples_file = 'dyer_beam/dev_pos_embeddings_beam=100.ptb_samples'
rnng_lstm_score_file = 'dyer_beam/dev_pos_embeddings_beam=100.ptb_samples.likelihoods'
rnng_gen_score_file = 'dyer_beam/dev_pos_embeddings_beam=100.samples.likelihoods2'


# In[2]:

# section 22, files in 
lstm_gold_file = 'wsj/dev_22.txt.stripped'
lstm_decode_file = 'expts/beam_size=50-at_word=5.decode_all'
lstm_stderr_file = 'expts/beam_size=50-at_word=5.stderr_all'


# In[3]:

import rnng_interpolate
import rnng_output_to_nbest
import rnng_threeway_interpolate
import decode_analysis


# In[4]:

from ptb_reader import get_tags_tokens_lowercase


# In[5]:

# RNNG order


# In[29]:

with open(rnng_gold_file) as f:
    rnng_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
with open(rnng_gold_file) as f:
    rnng_gold_trees = [line.strip() for line in f]


# In[7]:

with open(rnng_samples_file) as f:
    rnng_indices_discrim_scores_and_parses = list(rnng_output_to_nbest.parse_rnng_file(f))


# In[8]:

sample_lens = [len(l) for l in rnng_indices_discrim_scores_and_parses]


# In[9]:

with open(rnng_lstm_score_file) as f:
    rnng_lstm_scores = list(rnng_interpolate.parse_likelihood_file(f))


# In[10]:

with open(rnng_gen_score_file) as f:
    rnng_gen_scores = list(rnng_threeway_interpolate.parse_rnng_gen_score_file(f, sample_lens))


# In[11]:

# LSTM order


# In[12]:

with open(lstm_gold_file) as f:
    lstm_gold_tokens = [get_tags_tokens_lowercase(line)[1] for line in f]
with open(lstm_gold_file) as f:
    lstm_gold_trees = [line.strip() for line in f]


# In[13]:

lstm_decode_instances = decode_analysis.parse_decode_output_files(lstm_gold_trees, lstm_decode_file, lstm_stderr_file)


# In[14]:

rs = [tuple(s) for s in rnng_gold_tokens]
len(set(rs))


# In[15]:

ls = [tuple(s) for s in lstm_gold_tokens]
len(set(ls))


# In[16]:

import collections


# In[17]:

dups = collections.defaultdict(list)
for i, e in enumerate(ls):
    dups[e].append(i)
for k, v in sorted(dups.iteritems()):
    if len(v) >= 2:
        print '%s: %r' % (k, v)


# In[18]:

lstm_decode_instances[127].gold_linearized


# In[19]:

lstm_decode_instances[1056].gold_linearized


# In[20]:

rnng_indices_to_lstm_indices = [
    lstm_gold_tokens.index(rnng_gt) for rnng_gt in rnng_gold_tokens
]


# In[21]:

len(rnng_indices_to_lstm_indices)


# In[22]:

len(set(rnng_indices_to_lstm_indices))


# In[23]:

len(set(tuple(s) for s in lstm_gold_tokens) - set())


# In[24]:

# get max scoring from rnng samples under lstm probability


# In[34]:

best_from_proposal = [
    max((lstm_score, parse) for (_, _, parse), lstm_score in zip(ipp, scores))
    for ipp, scores in zip(rnng_indices_discrim_scores_and_parses, rnng_lstm_scores)
]
# for i, (ipp, lstm_scores) in enumerate(zip(rnng_indices_discrim_scores_and_parses, rnng_lstm_scores)):
        


# In[35]:

best_from_proposal[0]


# In[36]:

lstm_decode_instances[rnng_indices_to_lstm_indices[0]]


# In[37]:

best_proposal_fname = '/tmp/best_from_proposal.out'
best_proposal_gold_fname = '/tmp/best_from_proposal_and_gold.out'
best_proposal_decode_fname = '/tmp/best_from_proposal_and_decode.out'
best_proposal_gold_decode_fname = '/tmp/best_from_proposal_gold_and_decode.out'


# In[50]:

only_decode_fname = '/tmp/only_decode.out'


# In[39]:

di = lstm_decode_instances[0]


# In[40]:

di.gold_ptb


# In[46]:

N_sents = len(rnng_indices_discrim_scores_and_parses)


# In[49]:

import evaluate


# In[48]:

def percent_tuple_str(n):
    return "%s / %s (%0.2f%%)" % (n, N_sents, float(n) * 100 / N_sents)


# In[56]:

decode_beats_proposal = 0
decode_beats_gold = 0
decode_beats_gold_and_proposal = 0
# just the samples
with open(best_proposal_fname, 'w') as f_proposal,    open(best_proposal_gold_fname, 'w') as f_proposal_gold,    open(best_proposal_decode_fname, 'w') as f_proposal_decode,    open(best_proposal_gold_decode_fname, 'w') as f_proposal_gold_decode,    open(only_decode_fname, 'w') as f_decode:
    for i, (best_proposal_score, best_proposal) in enumerate(best_from_proposal):
        gold_parse = rnng_gold_trees[i]
        decode_instance = lstm_decode_instances[rnng_indices_to_lstm_indices[i]]
        assert(get_tags_tokens_lowercase(decode_instance.gold_ptb)[1] == get_tags_tokens_lowercase(gold_parse)[1])
#         print(i, best_proposal_score, decode_instance.pred_score, decode_instance.gold_score)
        f_proposal.write("%s\n" % best_proposal)
        
        f_decode.write("%s\n" % decode_instance.gold_ptb)
        
        if decode_instance.pred_score > best_proposal_score + 1e-3:
            decode_beats_proposal += 1
            
        if decode_instance.pred_score > decode_instance.gold_score + 1e-3:
            decode_beats_gold += 1
            
        if decode_instance.pred_score > decode_instance.gold_score + 1e-3 and decode_instance.pred_score > best_proposal_score + 1e-3:
            decode_beats_gold_and_proposal += 1
        
        f_proposal_decode.write("%s\n" % max([
                    (best_proposal_score, best_proposal),
                    (decode_instance.pred_score, decode_instance.pred_ptb)
                ], key=lambda t: t[0])[1])
        
        f_proposal_gold.write("%s\n" % max([
                    (best_proposal_score, best_proposal),
                    (decode_instance.gold_score, decode_instance.gold_ptb)
                ], key=lambda t: t[0])[1])
        
        f_proposal_gold_decode.write("%s\n" % max([
                    (best_proposal_score, best_proposal),
                    (decode_instance.pred_score, decode_instance.pred_ptb),
                    (decode_instance.gold_score, decode_instance.gold_ptb)
                ], key=lambda t: t[0])[1])
            
print("decode beats proposal:\t" + percent_tuple_str(decode_beats_proposal))
print("decode beats gold:\t" + percent_tuple_str(decode_beats_gold))
print("decode beats gold and prop:\t" + percent_tuple_str(decode_beats_gold_and_proposal))
print
print("gold sanity check (R, P, F1, exact match):")
print(evaluate.eval_b(rnng_gold_file, only_decode_fname))
print("rescore proposal (R, P, F1, exact match):")
print(evaluate.eval_b(rnng_gold_file, best_proposal_fname))
print("rescore proposal+gold (R, P, F1, exact match):")
print(evaluate.eval_b(rnng_gold_file, best_proposal_gold_fname))
print("rescore proposal+decode (R, P, F1, exact match):")
print(evaluate.eval_b(rnng_gold_file, best_proposal_decode_fname))
print("rescore proposal+decode+gold (R, P, F1, exact match):")
print(evaluate.eval_b(rnng_gold_file, best_proposal_gold_decode_fname))


# In[ ]:



