
#' @title bag_sampling_classification_dataset
#' @description Given labelled bags of instances, randomly generate new bags bag_size by random sampling random sample instances
#' @param h5f a character vector of h5 filenames (recycled to maximum length of the arguments)
#' @param name a character vector of array names in the h5 file (recycled to maximum length of the arguments)
#' @param perm a list of integer vectors of array permuations to apply (recylced if needed).
#' @param index a list of `alist()` for array subsetting
#' @param transform a unary function to be apply on each element of the dataset
#' @param transform_h5elt a unary function to be apply on the list of raw HDF5Array (e.g. permutation and subset indexing)
#' @import torch
#' @import tibble
#' @import rlang
#' @export
#' @examples
#' x <- Matrix(0,3000,34)
#' sample_ids <- gl(3,1000)
#' y <- c("a","b","a")
#' bag_sampling_classification_dataset(x,sample_ids,y,nbag=10,bag_size=100)[1:3]
bag_sampling_classification_dataset <- torch::dataset(
  name = "bag_sampling_dataset",

  initialize = function(x,sample_ids,y,nbag=1000L,bag_size=100L,seed=1234L,replace=TRUE,post_process=identity,balanced=TRUE) {
    sample_ids <- as.factor(sample_ids)
    y <- as.factor(y)
    stopifnot(length(sample_ids)==nrow(x))
    stopifnot(length(y)==nlevels(sample_ids))
    stopifnot(all(!is.na(y)))
    self$post_process <- post_process
    self$x <- x
    self$samples <- tibble(
      sample_id = levels(sample_ids),
      y = y,
      cells = split(seq(nrow(x)),sample_ids)
    )

    #-#-#-#-#-#-#-#-#
    # Build bags
    #-#-#-#-#-#-#-#-#

    # First assign one sample to each bag
    set.seed(seed)
    if (balanced) {
      self$bags <- self$samples |>
        select(sample_id,y) |>
        slice_sample(prop=1) |>
        group_by(y) |>
        reframe(
          i = seq(nbag),
          sample_id = rep(sample_id,length.out=nbag)
        ) |>
        arrange(i,y) |>
        slice_head(n=nbag) |>
        select(!c(i,y)) |>
        rowid_to_column("bag_id")
    } else {
      self$bags <- tibble(bag_id = seq(nbag)) %>%
        # randomly select a sample for each bag to draw from it
        mutate(sample_id = sample(rep(self$samples$sample_id,length.out = n())))
    }

    # Then randomly select cells from selected sample
    self$bags <- self$bags %>%
      inner_join(self$samples,by="sample_id",relationship="many-to-many") %>%
      mutate(cells = map(cells,sample,size=bag_size,replace=replace)) %>%
      mutate(cells = do.call(rbind,cells))
  },
  .length = function() {
    nrow(self$bags)
  },
  .getbatch = function(index) {
    B <- dplyr::slice(self$bags,index)
    list(
      x = self$x[B$cells,] |> as.matrix() |> torch_tensor() |> torch_reshape(c(dim(B$cells),ncol(self$x))),
      y = torch_tensor(B$y)
    ) |> self$post_process()
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)
