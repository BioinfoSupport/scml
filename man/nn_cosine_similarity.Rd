% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/torch_module.R
\name{nn_cosine_similarity}
\alias{nn_cosine_similarity}
\title{title nn_cosine_similarity}
\usage{
nn_cosine_similarity(in_features, out_features, eps = 1e-08, attention = FALSE)
}
\arguments{
\item{in_features}{Input number of features.}

\item{out_features}{Output number of features.}

\item{eps}{a positive scalar value to avoid divisions by 0}

\item{attention}{a logical, when true the module compute cos(w,|w|x), when FALSE it computes cos(w,x)}
}
\description{
A nn_cosine_similarity layer, compute cos(w,x) or cos(w,|w|x) when attention is TRUE
}
\examples{
p <- nn_cosine_similarity(10,5)(torch::torch_rand(32,10))
p <- nn_cosine_similarity(10,5,attention=TRUE)(torch::torch_rand(32,10))
}
