
test_that("module works", {
  cs <- nn_cell_scorer(1:4,c("type1","type2"))
  expect_no_error(cs(data.matrix(iris[1:4])))
})


