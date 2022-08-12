#!/bin/bash

export LD_LIBRARY_PATH=/GW/NIRKB/nobackup/Setup/miniconda3/envs/gpu-env/lib/

if [ $# == 0 ]; then
  method='eva_multi'
  export USE_KNRM=no
elif [ $1 == 'knrm' ]; then
  method='eva_multi_knrm'
  export USE_KNRM=yes
fi

python -m search.eva_multi ../evaluation/dl19.queries.tsv > ../evaluation/${method}/dl19.${method}.search.txt
python -m search.eva_multi ../evaluation/dl20.queries.tsv > ../evaluation/${method}/dl20.${method}.search.txt
python -m search.eva_multi ../evaluation/dlhard.queries.tsv > ../evaluation/${method}/dlhard.${method}.search.txt
python -m search.eva_multi ../evaluation/mmdev.queries.tsv > ../evaluation/${method}/mmdev.${method}.search.txt
