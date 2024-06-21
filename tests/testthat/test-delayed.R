
# test_that("delayed_dataset works", {
#   expect_no_error(delayed_dataset(x=iris[1:4],y=iris[1:2])[1:5])
#   expect_no_error(delayed_dataset(iris[1:4],iris[1:2])[1:5])
#   expect_no_error(delayed_dataset(x=iris[1:4],y=as.matrix(as.integer(iris$Species)),dtypes=list(torch_float(),torch_long()))[1:5])
# })
