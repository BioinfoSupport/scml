


#' @title delayed_classifier_dataset
#' @description A torch dataset which accepts arrays stored .
#' @param h5f a character vector of h5 filenames (recycled to maximum length of the arguments)
#' @param name a character vector of array names in the h5 file (recycled to maximum length of the arguments)
#' @param perm a list of integer vectors of array permuations to apply (recylced if needed).
#' @param index a list of `alist()` for array subsetting
#' @param transform a unary function to be apply on each element of the dataset
#' @param transform_h5elt a unary function to be apply on the list of raw HDF5Array (e.g. permutation and subset indexing)
#' @import torch
#' @import HDF5Array
#' @import rlang
#' @export
#' @examples
#' m <- HDF5Array::writeHDF5Array(array(runif(100*50*3),c(100,50,3)))
#' h5_dataset(m@seed@filepath,c(m@seed@name,m@seed@name))[1:3]
h5_dataset <- torch::dataset(
  name = "h5_dataset",
  initialize = function(h5f,name,transform=identity,transform_h5elt=identity) {
    self$h5f <- rep_len(h5f,max(length(h5f),length(name)))
    self$name <- rep_len(name,length(self$h5f))
    self$transform = transform
    self$transform_h5elt = transform_h5elt

    # Check dimensions are compatible
    self$num_elt <- local({
      A <- mapply(HDF5Array::HDF5Array,self$h5f,self$name,USE.NAMES=FALSE) |>
        self$transform_h5elt()
      num_elt <- sapply(A,nrow)
      stopifnot("All array must have same number of row"=(min(num_elt)==max(num_elt)))
      min(num_elt)
    })
  },
  .length = function() {self$num_elt},
  .getbatch = function(index) {
    A <- mapply(HDF5Array::HDF5Array,self$h5f,self$name,USE.NAMES=FALSE) |>
      self$transform_h5elt()
    lapply(A,\(a) {
      idx <- c(alist(a,index,drop=FALSE),rep_len(list(quote(expr=)),length(dim(a))-1L))
      torch_tensor(as.array(do.call(`[`,idx)))
    }) |>
      self$transform()
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)


