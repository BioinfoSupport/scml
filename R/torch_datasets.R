

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
#' @description A torch dataset which accepts DelayedArrays.
#' @param x Expression matrix.
#' @param y Cell type array.
#' @import torch
#' @import HDF5Array
#' @export
#' @examples
#' #x <- h5_dataset("../../data/ABAMouseTENx/sce/assays.h5",name.x="/assay001*")
h5_dataset <- torch::dataset(
  name = "h5_dataset",
  get_array = function(x,name,row_idx=NULL,col_idx=NULL) {
    if (!is.null(name)) {
      x <- HDF5Array::HDF5Array(self$x,sub("[*]$","",name))
      if (grepl("[*]$",name)) x <- t(x)
    }
    if (!is.null(row_idx)) x <- x[row_idx,,drop=FALSE]
    if (!is.null(col_idx)) x <- x[,col_idx,drop=FALSE]
    return(x)
  },
  initialize = function(x,y=NULL,name.x=NULL,name.y=NULL,row_idx=NULL,col_idx.x=NULL) {
    self$x <- x
    self$y <- y
    self$name.x <- name.x
    self$name.y <- name.y
    self$row_idx <- row_idx
    self$col_idx.x <- col_idx.x

    # Check x and y dimensions are compatible
    x <- self$get_array(self$x,self$name.x,self$row_idx,self$col_idx.x)
    if (!is.null(self$y)) {
      y <- self$get_array(self$y,self$name.y,self$row_idx)
      stopifnot("x and y are expected to have the same number of row" = nrow(x)==nrow(y))
    }
    self$num_elt <- nrow(x)
  },
  .length = function() {self$num_elt},
  .getbatch = function(index) {
    x <- self$get_array(self$x,self$name.x,self$row_idx,self$col_idx)
    x <- torch_tensor(as.matrix(x[index, , drop = FALSE]))
    if (is.null(self$y)) {
      return(list(x = x))
    } else {
      y <- self$get_array(self$y,self$name.y,self$row_idx)
      y <- torch_tensor(as.matrix(y[index, , drop = FALSE]))
      return(list(x = x, y = y))
    }
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

