


#' @title bag_sampling_dataset
#' @description Given labelled bags of instances, randomly generate new bags bag_size by random sampling random sample instances
#' @param x a generic 2D matrix object, with elements in rows and features as column
#' @param bags a list of integer vectors referencing rows of x
#' @param y a vector of target values for each bag, so the length must match nlevels(sample_ids)
#' @param nbag number of random bags to generate in the dataset
#' @param bag_size size of the bags to generate
#' @param seed set seed for random sampling
#' @param replace a logical, if TRUE the same elements might be samples multiple time.
#' @param post_process a function to use to post_process the batches
#' @param balanced a logical, if TRUE will generate the same number of bag for each target value
#' @import torch
#' @importFrom tibble tibble rowid_to_column
#' @importFrom dplyr mutate inner_join select slice_sample group_by ungroup slice summarize n if_else
#' @importFrom purrr map
#' @importFrom tidyr unnest
#' @export
#' @examples
#' bag_sampling_dataset(
#'   matrix((1:1000-1)%%100+1,100,10),
#'   split(seq(100),gl(20,5)),
#'   gl(2,10),
#'   nbag = 7L,bag_size=5L,balanced=TRUE
#' )[1:3]
bag_sampling_dataset <- torch::dataset(
  name = "bag_sampling_dataset",

  initialize = function(x,bags,y,nbag=1000L,bag_size=100L,seed=1234L,replace=TRUE,post_process=identity,balanced=TRUE) {
    stopifnot(identical(length(y),length(bags)))
    stopifnot("some integers in bags are out of range" = max(unlist(bags))<=nrow(x))
    stopifnot("some integers in bags are out of range" = min(unlist(bags))>=1L)
    stopifnot(all(!is.na(y)))
    self$post_process <- post_process
    self$x <- x
    self$input_bags <- tibble(
      input_bag_idx = seq_along(bags),
      y = y,
      elements = bags
    )

    #-#-#-#-#-#-#-#-#
    # Build bags
    #-#-#-#-#-#-#-#-#

    # First assign one sample to each bag
    set.seed(seed)
    if (balanced) {
      self$bags <- self$input_bags |>
        select(input_bag_idx,y) |>
        group_by(y) |>
        summarize(input_bag_idx = list(input_bag_idx)) |>
        ungroup() |>
        mutate(sz = nbag %/% n() + if_else(seq(n())<=(nbag %% n()),1L,0L)) |>
        mutate(input_bag_idx = map2(input_bag_idx,sz,~sample(.x,.y,replace=replace))) |>
        unnest(input_bag_idx) |>
        slice_sample(prop=1) |>
        select(!c(y,sz)) |>
        rowid_to_column("bag_id")
    } else {
      self$bags <- tibble(bag_id = seq(nbag)) %>%
        # randomly select a sample for each bag to draw from it
        mutate(input_bag_idx = sample(rep(self$input_bags$input_bag_idx,length.out = n())))
    }

    # Then randomly select elements from selected sample
    self$bags <- self$bags %>%
      inner_join(self$input_bags,by="input_bag_idx",relationship="many-to-many") %>%
      mutate(elements = map(elements,sample,size=bag_size,replace=replace))
  },
  .length = function() {
    nrow(self$bags)
  },
  .getbatch = function(index) {
    B <- dplyr::slice(self$bags,index)
    list(
      x = local({
        x <- self$x[unlist(B$elements,use.names = FALSE),] |>
          as.matrix() |>
          torch_tensor() |>
          torch_reshape(c(nrow(B),-1,ncol(self$x)))
      }),
      y = torch_tensor(B$y)
    ) |> self$post_process()
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)

