# EVA

## Method Description
EVA is a knowledge-enhanced dense retrieval method where we enrich dense representations with entity information from a knowledge source.
We devise a strategy for creating many entity representations which reflect multiple views of a document.
This supports us to maximize the chance of matching documents with potential future queries.

## Setup

Please set up Python 3.9 for EVA, additionally we need to install several dependencies as follows:

    git clone https://github.com/haidangtran1989/EVA.git
    cd EVA
    pip install -r requirements.txt

Subsequently, we need to download compressed file from ``https://nextcloud.mpi-klsb.mpg.de/index.php/s/4GwSErddGHxp8b4`` to the EVA folder.
This file should be uncompressed with the following command:

    tar xvzf all_data.tar.gz

After this, please make sure that we have the folders `data`, `evaluation`, `index` and `models` under the EVA folder.
Note that total hard drive volume requirement is at least 200 GiB as these folders contains pre-built indices for EVA methods.

## Usage
We measure the performance of EVA methods on `DL-19`, `DL-20`, `DL-HARD` and `MS-MARCO Dev` test sets.
The annotated queries of these datasets are stored in `evaluation` folder.
To perform dense retrieval on these queries, we can run the following command for EVA Multi:

    ./eva_multi_search.sh

We can add `knrm` parameter for EVA Multi-KNRM method:

    ./eva_multi_search.sh knrm

These two commands encode queries into representations and look for closest document representations from the pre-built indices.
The search results are stored in `eva_multi` and `eva_multi_knrm` sub-folders of `evaluation`.

To evaluate the search results, we could run the following commands:

    ./eva_multi_evaluator.sh
    ./eva_multi_evaluator.sh knrm

When this is done, please locate to `eva_multi` and `eva_multi_knrm` folders again. The evaluation results are presented in `evaluation.txt` files.
