#!/bin/bash

if [ $# == 0 ]; then
  method='eva_multi'
  export USE_KNRM=no
elif [ $1 == 'knrm' ]; then
  method='eva_multi_knrm'
  export USE_KNRM=yes
fi

python -m evaluation.evaluator ../evaluation/dl19.passage.judgements.txt ../evaluation/${method}/dl19.${method}.search.txt ../evaluation/${method}/dl19.${method}.evaluation.txt 2
python -m evaluation.evaluator ../evaluation/dl20.passage.judgements.txt ../evaluation/${method}/dl20.${method}.search.txt ../evaluation/${method}/dl20.${method}.evaluation.txt 2
python -m evaluation.evaluator ../evaluation/dlhard.passage.judgements.txt ../evaluation/${method}/dlhard.${method}.search.txt ../evaluation/${method}/dlhard.${method}.evaluation.txt 2
python -m evaluation.evaluator ../evaluation/mmdev.passage.judgements.txt ../evaluation/${method}/mmdev.${method}.search.txt ../evaluation/${method}/mmdev.${method}.evaluation.txt 1
