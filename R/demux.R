

#' @title demux
#' @description  Solve a least square fitting problem with the constraints that coefficient are positive and sum to 1.
#'
#' The solution is useful to demultiplex a multiplexed signal where component profiles are known.
#' For example to demultiplex a bulk RNA-seq signal into known underlying cell-type profiles.
#'
#' The function find the solution to the minimization problem:
#'   min_a (a.x - y)^2
#' s.t.
#'   a>0 and sum a=1
#' @param x numeric matrix (n x m) of m components
#' @param y a vector with multiplexed expression profile of length n
#' @return a vector of m positive coefficients summing to one.
#' @import LowRankQP
#' @export
#'
#' @examples
#' x <- matrix(runif(1000),ncol=5)
#' alpha_true <- c(0.3,0.1,0.1,0.4,0.1)
#' y <- x %*% alpha_true + rnorm(nrow(x),0,0.03)
#' plot(x %*% alpha_true,y,xlim=c(0,1),ylim=c(0,1));abline(0,1)
#' alpha_pred <- demux(x,y)
#' plot(x %*% alpha_pred,y,xlim=c(0,1),ylim=c(0,1));abline(0,1)
demux <- function(x,y) {
    x <- as.matrix(x)
    y <- as.vector(y)
    stopifnot(identical(nrow(x),length(y)))
    w <- LowRankQP::LowRankQP(method="LU",
        Vmat = crossprod(x),
        dvec = -colSums(y * x),
        Amat = matrix(1,1,ncol(x)),
        bvec = 1,
        uvec = rep(1,ncol(x))
    )
    as.vector(w$alpha)
}








