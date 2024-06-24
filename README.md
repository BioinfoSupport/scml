

An R package with a Deep Learning architecture efficent to train models on single-cell transcriptomic data.
In particular, contains an helper function to train and predict multi-class and ordinal regression models on single-cell data.


# Installation
```
devtools::install_github("BioinfoSupport/scml")
```


# Usage
```
# Warning: the function expect genes as column, so you usually need to transpose your expression matric
fm <- scml::train_delayed_classifier(
  cell_x_gene_matrix,                        # usually t(log2(RPM+1))
  cells_class,                               # a factor containing the class of the cells
  accelerator = luz::accelerator(cpu=TRUE),  # train on cpu
  input_dropout_rate = 0.75,                 # Amount of dropout during training
  batch_size = 512L,
  valid_data = 0.1,                          # Percentage of data used for validation
  pre_pruning_epoch = 5,                     # Number of epoch before pruning the model
  post_pruning_epoch = 5                     # Number of epoch after pruning the model
)
```
