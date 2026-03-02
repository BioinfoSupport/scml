

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


#' @title as_dataloader.list
#' @description Override generic function to handle list(x=DelayedArray,y=factor).
#' @param x the list to convert
#' @param ... additional arguments are passed to `as_dataloader`
#' @importFrom luz as_dataloader
#' @importFrom methods is
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

