

#' @title delayed_classifier_dataset
#' @description A torch dataset which accepts DelayedArrays.
#' @param x Expression matrix.
#' @param y Cell type array.
#' @param x_dtype torch types.
#' @param y_dtype torch types.
#' @import torch
#' @import DelayedArray
#' @export
delayed_classifier_dataset <- torch::dataset(
  name = "delayed_dataset",
  initialize = function(x, y = NULL, x_dtype = torch_float(), y_dtype = torch_long()) {
    stopifnot(identical(length(dim(x)), 2L))
    if (is.null(y)) y <- rep(0L, nrow(x))
    stopifnot(identical(length(y), nrow(x)))
    self$x <- x
    self$y <- y
    self$x_dtype <- x_dtype
    self$y_dtype <- y_dtype
  },
  .length = function() {
    length(self$y)
  },
  .getbatch = function(index) {
    list(
      x = torch_tensor(as.matrix(self$x[index, , drop = FALSE]), self$x_dtype),
      y = torch_tensor(self$y[index], self$y_dtype)
    )
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)



#' @title delayed_classifier_dataset
#' @description A torch dataset which accepts arrays stored .
#' @param h5f a character vector of h5 filenames (recycled to maximum length of the arguments)
#' @param name a character vector of array names in the h5 file (recycled to maximum length of the arguments)
#' @param perm a list of integer vectors of array permuations to apply (recylced if needed).
#' @param index a list of `alist()` for array subsetting
#' @import torch
#' @import HDF5Array
#' @import rlang
#' @export
#' @examples
#' m <- HDF5Array::writeHDF5Array(array(runif(100*50*3),c(100,50,3)))
#' h5_dataset(m@seed@filepath,c(m@seed@name,m@seed@name))[1:3]
h5_dataset <- torch::dataset(
  name = "h5_dataset",
  get_array = function(h5f,name,perm,index) {
    x <- HDF5Array::HDF5Array(h5f,name)
    if (!is.null(perm)) x <- aperm(x,perm)
    x <- inject(x[!!!index,drop=FALSE])
    x
  },
  initialize = function(h5f,name,perm=list(),index=list()) {
    rlang::inform("h5_dataset is experimental and not well tested.")
    self$h5f <- rep_len(h5f,max(length(h5f),length(name),length(perm),length(index)))
    self$name <- rep_len(name,length(self$h5f))
    self$index <- rep_len(index,length(self$h5f))
    self$perm <- rep_len(perm,length(self$h5f))

    # Check dimensions are compatible
    self$num_elt <- mapply(self$get_array,self$h5f,self$name,self$perm,self$index) |>
      sapply(nrow)
    stopifnot("All array must have same number of row"=(min(self$num_elt)==max(self$num_elt)))
    self$num_elt <- min(self$num_elt)
  },
  .length = function() {self$num_elt},
  .getbatch = function(index) {
    A <- mapply(self$get_array,self$h5f,self$name,self$perm,self$index)
    lapply(A,\(a){
      idx <- c(alist(a,index,drop=FALSE),rep_len(list(quote(expr=)),length(dim(m))-1L))
      torch_tensor(as.array(do.call(`[`,idx)))
    })
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)




#' @title as_dataloader.list
#' @description Override generic function to handle list(x=DelayedArray,y=factor).
#' @param x the list to convert
#' @param ... additional arguments are passed to `as_dataloader`
#' @import luz
#' @import DelayedArray
#' @export
as_dataloader.list <- function(x,...) {
  if ((length(x)>=1L) && (is(x[[1L]],"DelayedArray") || is(x[[1L]],"Matrix"))) {
    if (length(x)==1L) {
      as_dataloader(delayed_classifier_dataset(x[[1L]]),...)
    } else if ((length(x)==2L) && (is.factor(x[[2L]]) || is.integer(x[[2L]]))) {
      as_dataloader(delayed_classifier_dataset(x[[1L]],x[[2L]]),...)
    } else {
      NextMethod("as_dataloader")
    }
  } else {
    NextMethod("as_dataloader")
  }
}

