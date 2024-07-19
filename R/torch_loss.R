
#' @title nnf_ordinal_regression_loss
#' @description  Ordinal regression loss function.
#' @param pred Predicted values.
#' @param target Target values.
#' @param C a square 2D float tensor representing the cost matrix.
#' @param margin a scalar positive value.
#' @return loss value
#' @export
#' @import torch
nnf_ordinal_regression_loss <- function(pred,target,C,margin=1.0) {
  pred <- pred$flatten()
  target <- target$flatten()
  stopifnot(identical(ncol(C),nrow(C)))

  ij <- torch_combinations(torch_arange(1L,length(pred), dtype = torch_long()))
  y <- target[ij]
  l <- nnf_margin_ranking_loss(
    input1 = pred[ij[,1L]],
    input2 = pred[ij[,2L]],
    target = sign(y[,1L] - y[,2L]),
    reduction = "none",margin = margin
  )
  w <- torch_gather(C[y[,1L],],2L,y[,2L,drop=FALSE])$squeeze(-1L)
  l$dot(w) / torch_maximum(sum(w!=0),1L)
}

#' @title nn_ordinal_regression_loss
#' @description  Ordinal regression loss module
#' @param C a square 2D float tensor representing the cost matrix.
#' @export
#' @import torch
#' @examples
#' nn_ordinal_regression_loss(1 - torch::torch_eye(3L))
nn_ordinal_regression_loss <- torch::nn_module(
  "nn_ordinal_regression_loss",
  initialize = function(C) {
    self$C <- C
  },
  forward = function(input, target){
    nnf_ordinal_regression_loss(input, target, C = self$C)
  }
)





#' @title nnf_dag_loss
#' @description Direct Acyclic Graph loss function.
#' @param pred Predicted values.
#' @param target Target values.
#' @param e a 2 column integer matrix (edge list of the DAG), with parent index
#'          in the first column and child index in the second
#' @param w weight vector for each node of the dag (how much to penalize a node)
#' @param margin a scalar positive value.
#' @return loss value
#' @export
#' @import torch
nnf_dag_loss <- function(pred,target,e,w,margin=1.0) {
  #e <- matrix(ncol = 2,byrow = TRUE,c(4L,5L,5L,1L,5L,2L,4L,3L)) |> torch_tensor()
  #w <- torch_ones(5L);pred <- torch_zeros(32,5L);target <- torch_randint(0L,2L,c(32L,5L));
  e <- torch_tensor(e)
  w <- torch_tensor(w)
  stopifnot(ncol(e)==2L)
  y2 <- target$index_select(-1L,e[,2L])  # y_true of the child
  f2 <- pred$index_select(-1L,e[,2L])  # child value
  ub <- torch_full_like(pred,+Inf)$index_reduce(-1L,e[,1L],f2$masked_fill(!y2,+Inf),"amin",include_self = FALSE)
  lb <- torch_full_like(pred,-Inf)$index_reduce(-1L,e[,1L],f2$masked_fill(y2,-Inf),"amax",include_self = FALSE)
  l <- (lb + margin - ub)$clamp_min(0)

  #nub <- torch_zeros_like(pred)$index_add(-1L,e[,1L],y2*w[e[,2L]])
  #nlb <- torch_zeros_like(pred)$index_add(-1L,e[,1L],(!y2)*w[e[,2L]])

  l$mv(w)$mean()
}


#' @title nn_dag_loss
#' @description Direct Acyclic Graph loss module
#' @param e a 2 column integer matrix (edge list of the DAG), with parent index
#'          in the first column and child index in the second
#' @param w weight vector for each node of the dag (how much to penalize a node)
#' @export
#' @import torch
nn_dag_loss <- torch::nn_module(
  "nn_ordinal_regression_loss",
  initialize = function(e,w) {
    self$e <- e
    self$w <- w
  },
  forward = function(input, target){
    nnf_dag_loss(input, target, e = self$e, w = self$w)
  }
)
