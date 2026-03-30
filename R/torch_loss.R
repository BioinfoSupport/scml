
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




#' @title nnf_hinge_loss
#' @description Hinge loss
#' @param input tensor of predicted values.
#' @param target tensor of binary target -1 or +1
#' @param margin a scalar positive value.
#' @param reduction name of the reduction to apply
#' @return hinge loss scalar value
#' @export
#' @import torch
nnf_hinge_loss <- function(input,target,margin=1.0,reduction=c("mean","sum","none")) {
  l <- torch_clamp(margin - input$flatten()*target,min=0.0)
  switch(match.arg(reduction),
    "mean" = l$mean(),
    "sum"  = l$sum(),
    "none" = l
  )
}


#' @title nn_hinge_loss
#' @description Hinge regression loss module computing
#' @param margin a scalar positive value.
#' @param reduction name of the reduction to apply
#' @export
#' @import torch
#' @examples
#' nn_hinge_loss(reduction="none")(torch_tensor(c(-2,-0.5,+0.7,3)),torch_tensor(c(-1,-1,+1,+1)))
nn_hinge_loss <- torch::nn_module(
  "nn_hinge_loss",
  initialize = function(margin=1.0,reduction="mean") {
    self$margin <- margin
    self$reduction = reduction
  },
  forward = function(input, target){
    nnf_hinge_loss(input, target, margin = self$margin, reduction = self$reduction)
  }
)




#' @title nnf_dag_loss
#' @description Direct Acyclic Graph loss function.
#' @param pred (m*n) float tensor of predicted values, where m is number of
#'    sample in the batch and n is number of node in the dag.
#' @param target a (m*n) binary tensor of target classes.
#' @param dag a 2 column integer matrix (edge list of the DAG), with parent index
#'    in the first column and child index in the second.
#' @param w float tensor of weights compatible with (m*n) tensor (how much to
#'    penalize a sample/node)
#' @param margin a scalar positive value.
#' @return loss value
#' @export
#' @import torch
nnf_dag_loss <- function(pred,target,dag,w=1,margin=1) {
  #dag <- matrix(ncol = 2,byrow = TRUE,c(4L,5L,5L,1L,5L,2L,4L,3L)) |> torch_tensor()
  #pred <- torch_rand(150,5L);target <- torch_randint(0L,2L,c(150L,5L));w <- 1
  dag <- torch_tensor(dag)
  w <- w$broadcast_to(dim(pred))
  stopifnot(ncol(dag)==2L)
  y2 <- target$index_select(-1L,dag[,2L]) # y_true of the child
  f2 <- pred$index_select(-1L,dag[,2L])   # child value
  ub <- torch_full_like(pred,+Inf)
  ub$index_reduce_(-1L,dag[,1L],f2$masked_fill(!y2,+Inf),"amin",include_self = FALSE)
  lb <- torch_full_like(pred,-Inf)
  lb$index_reduce_(-1L,dag[,1L],f2$masked_fill(y2,-Inf),"amax",include_self = FALSE)
  l <- (lb + margin - ub)$clamp_min(0)
  (l*w)$mean()
}


#' @title nn_dag_loss
#' @description Direct Acyclic Graph loss module
#' @param dag a 2 column integer matrix (edge list of the DAG), with parent index
#'          in the first column and child index in the second
#' @param w (m*n) float tensor of weights for each sample and node in the dag
#' @export
#' @import torch
nn_dag_loss <- torch::nn_module(
  "nn_dag_loss",
  initialize = function(dag,w=1) {
    self$dag <- dag
    self$w <- w
  },
  forward = function(input, target){
    nnf_dag_loss(input, target, dag = self$dag, w = self$w)
  }
)
