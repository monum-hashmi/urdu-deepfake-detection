"""Bootstrap statistics used in the paper.

The paper's confidence intervals and paired tests are CLUSTERED: whole clusters are resampled, where a
cluster holds a bona fide recording, its MP3 copy and every fake synthesised from its transcript
(key = source speaker + utterance number). 2,000 draws, paired one-sided test against ssl-xlsr_noaug,
each draw averaging the EER difference over the three seed pairs, Holm correction over all comparisons.
That implementation lives in review_analysis.py; this file only runs it, so there is one method.
A coarser, prompt-level clustering (utterance number across all speakers) is in review2_analysis.py.

An earlier version of this script resampled individual utterances within each class (1,000 draws).
It was replaced because it ignores the dependence between a recording and the fakes made from it;
its outputs (results/bootstrap_ci.csv, results/bootstrap_paired.csv) were removed.

Run:  python bootstrap.py        (same as: python review_analysis.py --n 2000)
"""
import sys

import review_analysis

if __name__ == "__main__":
    if "--n" not in sys.argv:
        sys.argv += ["--n", "2000"]
    review_analysis.main()
