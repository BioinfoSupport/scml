

#' @title predict_delayed
#' @description Predict cell type given a torch model.
#' @param N expression matrix, ideally log1p(RPM/100)/log(2) counts, but raw counts seems to work too.
#' @param luz_model fitted model to use for predictions
#' @param ignore.case if TRUE ignore case when aligning feature names (=gene names). It is useful for easy Human-Mouse gene-name matching.
#' @param batch_size Size of data loaded at training.
#' @param ... Additional argument to be passed to predict function.
#' @return A prediction matrix giving each input cell a score for each cell type.
#' @importFrom stats predict
#' @export
predict_delayed <- function(N, luz_model, ignore.case = FALSE, batch_size = 1024L, ...) {
  # Align the input matrix to desired matrix format, adding 0 when genes are missing
  if (isTRUE(ignore.case)) {
    i <- match(toupper(luz_model$model$feature_names), toupper(colnames(N)), nomatch = 0)
  } else {
    i <- match(luz_model$model$feature_names, colnames(N), nomatch = 0)
  }
  stopifnot(mean(i > 0) >= 0.2) # check that 20% of model genes are in the input matrix
  m <- cbind(N[, 1, drop = FALSE] * 0, N)[, i + 1]

  # Compute prediction by batch
  pred <- predict(
    luz_model,
    delayed_classifier_dataset(m),
    dataloader_options = list(batch_size = batch_size),
    ...
  )
  pred <- as_array(pred$to(device="cpu"))
  colnames(pred) <- luz_model$model$class_names
  pred
}




#' @title train_delayed_classifier
#' @description Allows user-friendly training set up of a torch model.
#' @param x Count matrix.
#' @param y Array containing the cell type class label.
#' @param pre_pruning_epoch Number of epoch before pruning
#' @param post_pruning_epoch Number of epoch after pruning. Set to 0
#' @param batch_size Size of batch of data loaded at each training step.
#' @param prune_size number of top/down gene to keep for each node of layer1 when pruning
#' @param lr_start initial learning rate (both at epoch 1 and after pruning)
#' @param lr_end final learning rate (both at pruning and at last epoch)
#' @param initial_weights_path a character to specify a file to load initial weights from.
#'                             Set to NA to disable.
#' @param autoresume_path a character to specify a file to store model weights at each epoch.
#'                        Set to NA to disable.
#' @param weight_decay value passed to the optim_adam optimizer.
#' @param input_dropout_rate Dropout rate to apply on input values when training
#' @param n A numeric integer vector of length>0 specifying the number of feature in each internal layer.
#' @param dropout_rates A numeric vector of length>0 of dropouts to apply before each internal layer.
#' @param y_type type of target factor: either nominal or ordinal (default: nominal)
#' @param C the cost matrix to use in ordinal regression
#' @param callbacks a list of additional callbacks to use
#' @param attention wether to use attention in the first layer (cosine_similarity layer)
#' @param ... additional parameters are passed to fit() call
#' @importFrom stats stepfun
#' @importFrom luz fit set_opt_hparams set_hparams setup luz_metric_accuracy luz_callback_lr_scheduler luz_callback_auto_resume luz_load_model_weights
#' @export
train_delayed_classifier <- function(
    x, y, pre_pruning_epoch = 25L, post_pruning_epoch = 25L,
    batch_size = 256L, prune_size = pmin(50L, ncol(x)),
    lr_start = 1e-3, lr_end = 1e-5,attention=TRUE,
    initial_weights_path = NA, autoresume_path = NA,
    n = c(4096L,256L), dropout_rates = 0.25,
    weight_decay = 0, input_dropout_rate = 0.5, y_type=c("nominal","ordinal"),C=NULL,
    callbacks=NULL,...)
{
  y <- as.factor(y)
  y_type <- match.arg(y_type)

  switch(y_type,
    nominal = {
      if (!is.null(C)) stop("Cost matrix not implemented yet for nominal target.")
      loss <- nn_multi_margin_loss(margin = 1)
      class_names <- levels(y)
      metrics <- list(luz_metric_accuracy())
    },
    ordinal = {
      if (is.null(C)) {
        C <- 1 - torch_eye(nlevels(y))
      }
      loss <- nn_ordinal_regression_loss(C = C)
      class_names <- "ordinal_score"
      metrics <- NULL
    }
  )

  fm <- nn_cell_scorer |>
    setup(
      optimizer = optim_adam,
      metrics = metrics,
      loss = loss
    ) |>
    set_hparams(feature_names = colnames(x), class_names = class_names,input_dropout_rate = input_dropout_rate, n = n, dropout_rates = dropout_rates, attention = attention) |>
    set_opt_hparams(weight_decay = weight_decay, lr = 1) |>
    fit(
      epoch = pre_pruning_epoch + post_pruning_epoch,
      data = delayed_classifier_dataset(x, as.integer(y)),
      dataloader_options = list(batch_size = batch_size, drop_last = FALSE, shuffle = TRUE),
      ...,
      callbacks = c(
        list(
          luz_callback_lr_geometric(c(pre_pruning_epoch,post_pruning_epoch),lr_start,lr_end)
        ),
        if (is.na(initial_weights_path)) NULL else luz_callback(initialize=function(){},on_fit_begin=function() {luz_load_model_weights(ctx,initial_weights_path)})(),
        if (post_pruning_epoch<=0L) NULL else luz_callback_prune("cosine.weight", n = prune_size, at_epoch = pre_pruning_epoch),
        if (is.na(autoresume_path)) NULL else luz_callback_auto_resume(path=autoresume_path),
        callbacks
      )
    )
  fm
}






