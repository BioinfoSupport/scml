test_that("ordinal regression works", {

  # Test with 2D predictions and targets
  C <- 1 - torch_eye(3)
  pred <- torch_tensor(matrix(c(3,4,5,6),ncol=1))
  target <- torch_tensor(matrix(c(1L,1L,3L,2L),ncol=1))
  loss <- nnf_ordinal_regression_loss(pred,target,C)
  expect_true(abs(loss$item() - 0.4) <= 1e-6)

  # Test with 1D predictions and targets
  pred <- pred$squeeze()
  target <- target$squeeze()
  loss <- nnf_ordinal_regression_loss(pred,target,C)
  expect_true(abs(loss$item() - 0.4) <= 1e-6)
})
