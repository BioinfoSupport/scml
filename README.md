
# scml

`scml` is a R package to build AI models on single-cell transcriptomic data
and predict/annotate new data with these models. The models typically takes as 
input `logcounts` expression matrix and train a multi-class classifier able to 
assign a class (e.g. cell-type) to each cell. Efficiency of the model lies in 
its architecture and in its training process.



| ![](man/figures/nn_arch.png) |
|:--:|
| **Figure 1:** *Overview of model architecture and training procedure used in `scml`* |

**Figure 1** illustrates the architecture of the model used by `scml`.
The specificities of the model are:

1- `scml` use cosine similarity in the first layer to normalize input data 
    combined with an attention mechanism.

2- The training procedure is done in two phases, with a weight pruning on the 
   first layer at end the first phase. The pruning keep for each cosine node 
   50 inputs with highest weights and 50 inputs with lowest weights.

3- In each phase, the learning rate parameter is decreased at each epoch 
   according to a geometric serie.

4- Depending on the prediction target, `scml` optimize `multi_margin_loss()` 
   (e.g. to predict cell-type) or `ordinal_regression_loss()` (e.g. to predict
   age of the organism the cell is coming from).


## Installation

You can install the development version of scml from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("BioinfoSupport/scml")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(scml)
fm <- train_delayed_classifier(
  data.matrix(iris[1:4]),
  iris$Species,
  pre_pruning_epoch = 100L,post_pruning_epoch = 100L,
  batch_size = 150L,
  valid_data = 0.1,
  input_dropout_rate = 0.1,
  accelerator = accelerator(cpu=TRUE)
)
```


## Model detail






