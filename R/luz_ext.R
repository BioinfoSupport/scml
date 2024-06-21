

#' @title luz_callback_prune
#' @param param Model parameters
#' @param n number of top/bottom features to keep after pruning
#' @param at_epoch Epoch at which compute pruning mask.
#' @import torch
#' @importFrom luz luz_callback
#' @export
luz_callback_prune <- luz_callback(
  initialize = function(param,n,at_epoch) {
    self$param <- param
    self$n <- n
    self$at_epoch <- at_epoch
  },
  on_train_batch_after_step = function() {
    if (ctx$epoch >= self$at_epoch) {
      w <- ctx$model$parameters[[self$param]]
      if (is.null(self$prune_mask)) {
        lo <- w$topk(self$n,largest = FALSE)[[1]][,self$n,drop=FALSE]
        hi <- w$topk(self$n,largest = TRUE)[[1]][,self$n,drop=FALSE]
        self$prune_mask <- (w>lo) & (w<hi)
      }
      w[self$prune_mask] <- 0
    }
  }
)


#' @title luz_callback_lr_geometric
#' @param epochs a vector of positive integers, defining a consecutive number of
#'        epochs.
#' @param lr_start a vector of initial learning rate which will be recycled to
#'        length(epochs)
#' @param lr_end a vector of ending learning rate which will be recycled to
#'        length(epochs).
#' @import torch
#' @importFrom luz luz_callback
#' @importFrom rlang inform
#' @importFrom glue glue
#' @export
luz_callback_lr_geometric <- luz_callback(
  initialize = function(epochs,lr_start=1e-3,lr_end=1e-6) {
    lr_start <- rep(lr_start,length.out=length(epochs))
    lr_end <- rep(lr_end,length.out=length(epochs))
    alpha <- (lr_end/lr_start)^(1/(epochs-1))
    self$lr <- rep(lr_start,epochs) * rep(alpha,epochs)^(sequence(epochs)-1)
  },
  on_epoch_begin = function() {
    epoch <- ctx$epoch
    epoch <- pmax(epoch,1L)
    epoch <- pmin(epoch,length(self$lr))
    rlang::inform(glue::glue("Update LR to: {self$lr[epoch]}"))
    for(i in seq_along(ctx$optimizers)) {
      for(j in seq_along(ctx$optimizers[[i]]$param_groups)) {
        ctx$optimizers[[i]]$param_groups[[j]]$lr <- self$lr[epoch]
      }
    }
  }
)
