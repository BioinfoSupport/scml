
test_that("nn_dag_loss works", {
  e <- matrix(ncol = 2,byrow = TRUE,c(
    4L,5L,
    5L,1L,
    5L,2L,
    4L,3L
  )) |> torch_tensor()
  expect_no_error({nn_dag_loss(e,w)(torch_zeros(32,5L),torch_ones(32,5L))})
})




