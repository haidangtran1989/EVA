if [ $# == 0 ]; then
  export USE_KNRM=no
elif [ $1 == 'knrm' ]; then
  export USE_KNRM=yes
fi

python -m learning_to_rank.train_eva_model ../data/triples.train.annotation.txt.gz
