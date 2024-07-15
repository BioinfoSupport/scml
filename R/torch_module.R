




#' @title title nn_cosine_similarity
#' @description A nn_cosine_similarity layer, compute cos(w,x) or cos(w,|w|x) when attention is TRUE
#' @param in_features Input number of features.
#' @param out_features Output number of features.
#' @param eps a positive scalar value to avoid divisions by 0
#' @param attention a logical, when true the module compute cos(w,|w|x), when FALSE it computes cos(w,x)
#' @import torch
#' @export
#' @examples
#' p <- nn_cosine_similarity(10,5)(torch::torch_rand(32,10))
#' p <- nn_cosine_similarity(10,5,attention=TRUE)(torch::torch_rand(32,10))
nn_cosine_similarity <- nn_module(
  inherit = nn_linear,
  initialize = function(in_features,out_features,eps=1e-8,attention=FALSE) {
    super$initialize(in_features = in_features, out_features = out_features,bias = FALSE)
    self$eps <- eps
    self$attention <- attention
  },
  forward = function(x) {
    w_norm <- self$weight$norm(dim=-1L,p=2,keepdim = TRUE)$clamp_min(self$eps)
    if (self$attention) {
      wx_norm <- nnf_linear(x,self$weight)$norm(dim=-1L,p = 2,keepdim = TRUE)$clamp_min(self$eps)
      return(nnf_linear(x/wx_norm,self$weight*self$weight$abs()/w_norm)$clamp(-1,1))
    } else {
      x_norm <- x$norm(dim=-1L,p = 2,keepdim = TRUE)$clamp_min(self$eps)
      return(nnf_linear(x/x_norm,self$weight/w_norm)$clamp(-1,1))
    }
  }
)



#' @description Cell type classifier NN-based.
#' @title nn_cell_scorer
#' @param feature_names Names of the features.
#' @param class_names Class of the cells.
#' @param input_dropout_rate Dropout rate to apply on input values when training
#' @param n A numeric integer vector of length>0 specifying the number of feature in each internal layer.
#' @param dropout_rates A numeric vector of length>0 of dropouts to apply before each internal layer.
#' @param ... additional parameters are passed to nn_linear_rescale() when creating first layer.
#' @import torch
#' @export
#' @examples
#' nn_cell_scorer(1:4,1:3)(torch::torch_rand(32,4))
nn_cell_scorer <- torch::nn_module(
  inherit = nn_sequential,
  initialize = function(feature_names,class_names,input_dropout_rate=0.5,n=256L,dropout_rates=0.25,...) {
    super$initialize()

    # Process parameters
    self$feature_names <- as.character(feature_names)
    self$class_names <- as.character(class_names)
    n <- as.integer(n)
    stopifnot("n must be a non-empty integer vector"=length(n)>0L)
    stopifnot("dropout_rates must be a non-empty numeric vector"=length(dropout_rates)>0L)
    dropout_rates <- as.numeric(dropout_rates)
    dropout_rates <- rep_len(dropout_rates,length(n)) # Recycle to length(n)

    # First layer
    self$add_module("input_dropout",nn_dropout(input_dropout_rate))
    self$add_module("cosine",nn_cosine_similarity(in_features = length(self$feature_names),out_features = n[1L],...))

    # Internal layers
    for(i in seq_along(n)[-1L]) {
      self$add_module(paste0(i,":dropout"),nn_dropout(dropout_rates[i-1L]))
      self$add_module(paste0(i,":linear"),nn_linear(in_features = n[i-1L], out_features = n[i]))
      self$add_module(paste0(i,":relu"),nn_relu())
    }

    # Last layer
    self$add_module("final_dropout",nn_dropout(dropout_rates[length(dropout_rates)]))
    self$add_module("final_linear",nn_linear(in_features = n[length(n)], out_features = length(self$class_names)))
  }
)



