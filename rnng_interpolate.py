from __future__ import print_function
import sys
from rnng_output_to_nbest import parse_rnng_file

def parse_likelihood_file(line_iter):
    sent_liks = []
    num_liks = None
    remaining = None
    sent_count = 0
    for line in line_iter:
        if remaining is None or remaining == 0:
            num_liks = int(line.strip())
            remaining = num_liks
            sent_count += 1
        else:
            remaining -= 1
            sent_liks.append(float(line.strip()))
            if remaining == 0:
                assert(len(sent_liks) == num_liks)
                yield sent_liks
                sent_liks = []
    assert(remaining == 0)

if __name__ == "__main__":
    rnng_output_file = sys.argv[1]
    likelihood_file = sys.argv[2]
    if len(sys.argv) >= 4:
        lmbda = float(sys.argv[3])
    else:
        lmbda = 1.0

    with open(rnng_output_file) as f_rnng:
        parses_by_sent = list(parse_rnng_file(f_rnng))

    with open(likelihood_file) as f_lik:
        likelihoods_by_sent = list(parse_likelihood_file(f_lik))

    assert(len(parses_by_sent) == len(likelihoods_by_sent))
    for parses, likelihoods in zip(parses_by_sent, likelihoods_by_sent):
        assert(len(parses) == len(likelihoods))
        (ix, proposal_score, parse), rescore = max(zip(parses, likelihoods),
                     key=lambda ((ix, proposal_score, parse), rescore): (1 - lmbda) * proposal_score + lmbda * rescore)
        print(parse)
