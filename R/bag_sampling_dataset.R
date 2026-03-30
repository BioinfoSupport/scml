


#' @title bag_sampling_classification_dataset
#' @description Given labelled bags of instances, randomly generate new bags bag_size by random sampling random sample instances
#' @param x a generic 2D matrix object, with elements in rows and features as column
#' @param sample_ids a factor whose length match nrow(x) defining bag of elements
#' @param y a vector of target values for each bag, so the length must match nlevels(sample_ids)
#' @param nbag number of random bags to generate in the dataset
#' @param bag_size size of the bags to generate
#' @param seed set seed for random sampling
#' @param replace a logical, if TRUE the same elements might be samples multiple time.
#' @param post_process a function to use to post_process the batches
#' @param balanced a logical, if TRUE will generate the same number of bag for each target value
#' @import torch
#' @importFrom tibble tibble rowid_to_column
#' @importFrom dplyr mutate inner_join select slice_sample slice_head arrange group_by reframe slice
#' @importFrom purrr map
#' @export
#' @examples
#' bags <- bag_sampling_classification_dataset(
#'   matrix((1:1000-1)%%100+1,100,10),
#'   gl(20,5),
#'   gl(2,10),
#'   nbag = 7L,bag_size=5L,balanced=TRUE
#' )[1:3]
bag_sampling_classification_dataset <- torch::dataset(
  name = "bag_sampling_dataset",

  initialize = function(x,sample_ids,y,nbag=1000L,bag_size=100L,seed=1234L,replace=TRUE,post_process=identity,balanced=TRUE) {
    sample_ids <- as.factor(sample_ids)
    stopifnot(identical(length(sample_ids),nrow(x)))
    stopifnot(identical(length(y),nlevels(sample_ids)))
    stopifnot(all(!is.na(y)))
    self$post_process <- post_process
    self$x <- x
    self$samples <- tibble(
      sample_id = levels(sample_ids),
      y = y,
      cells = split(seq(nrow(x)),sample_ids)
    )

    #-#-#-#-#-#-#-#-#
    # Build bags
    #-#-#-#-#-#-#-#-#

    # First assign one sample to each bag
    set.seed(seed)
    if (balanced) {
      self$bags <- self$samples |>
        select(sample_id,y) |>
        slice_sample(prop=1) |> # randomize
        group_by(y) |>
        reframe(
          i = seq(nbag),
          sample_id = rep(sample_id,length.out=nbag)
        ) |>
        arrange(i,y) |>
        slice_head(n=nbag) |>
        select(!c(i,y)) |>
        rowid_to_column("bag_id")
    } else {
      self$bags <- tibble(bag_id = seq(nbag)) %>%
        # randomly select a sample for each bag to draw from it
        mutate(sample_id = sample(rep(self$samples$sample_id,length.out = n())))
    }

    # Then randomly select cells from selected sample
    self$bags <- self$bags %>%
      inner_join(self$samples,by="sample_id",relationship="many-to-many") %>%
      mutate(cells = map(cells,sample,size=bag_size,replace=replace)) %>%
      mutate(cells = do.call(rbind,cells))
  },
  .length = function() {
    nrow(self$bags)
  },
  .getbatch = function(index) {
    B <- dplyr::slice(self$bags,index)
    list(
      x = local({
        x <- self$x[B$cells,] |> as.matrix()
        dim(x) <- c(dim(B$cells),ncol(self$x))
        torch_tensor(x)
      }),
      y = torch_tensor(B$y)
    ) |> self$post_process()
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)

