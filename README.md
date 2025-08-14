

# TAMLEC, an Extreme multi-label classification/completion algorithm.

**Authors:** Julien Audiffren, Christophe Broillet 

## Introduction
This repository contains the implementation of the TAMLEC algorithm, described in the paper [[1]](#1).

Examples datasets (MAG-CS, EURLex and PubMed) can be downloaded [here](https://github.com/eXascaleInfolab/HECTOR).

## How to run
To handle the packages dependencies and requirements, an `environment.yml` file is given. A [conda environment](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) can thus be created with
```
conda env create -f environment.yml
``` 
Afterwards the virtual environment is activated using
```
conda activate xmlc
```




TAMLEC also requires the *GloVe* word embedding. We used the `GloVe.840B.300d` version, which can be downloaded in the [official website](https://nlp.stanford.edu/projects/glove/). The downloaded file must placed in a `.vector_cache` directory by default. The location of the pre-trained word embeddings, a parameter named `path_to_glove`, can be modified in the `algorithms/tamlec.py` files.


Then, a config file needs to be created inside the `configs` folder, and updated with the desired parameters, see `configs/base_config.py` for an example. Some hard-coded parameters, such as the batch size for the different methods, can be modified at the top of the `misc/experiment.py` file. The framework is then launched and executed using
```
python configs/{name_of_the_config_file}.py
```


## Config options
This section details the various available options from the `base_config.py` file.
- **dataset_path**: Path to the dataset
- **output_path**: Path to the output directory folder (automatically created if not existing). The experiment folder will have the name `exp_name` inside the `output_path` (see below)
- **exp_name**: Name of the experiment, takes the name of the config file by default, better left unchanged to avoid name conflicts
- **device**: Device on which to run the experiment, either `cpu`, `cuda`, or `cuda:x` with `x` a specified GPU number
- **learning_rate**: Learning rate to use for training
- **seq_length**: Length of the input sequence, i.e. the number of tokens in one input sample
- **voc_size**: (Maximum) Size of the vocabulary
- **tokenization_mode**: What method to use to tokenize the texts, either `word`, `bpe`, `unigram`
- **k_list**: List of *@k* on which to evaluate the metrics for the final evaluation
- **k_list_eval_perf**: List of *@k* on which to evaluate the metrics during the training process
- **tamlec_params**: Various parameters for TAMLEC method
- **fewshot_exp**: If we want to run the few-shot experiments

Other, specific algorithms parameters, such as in `fastxml` and `parabel`, can be modified for the given method under the `algorithms` folder.

## References
<a id="1">[1]</a>
J. Audiffren, C. Broillet, L. Dolamic and P. Cudre-Mauroux (2025). Extreme Multi-Label Completion for Semantic Document Tagging with Taxonomy-Aware Parallel Learning. CIKM 2025.
