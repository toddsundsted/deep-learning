require "mxnet"

MXNet::Random.seed(1)

10.times do |i|
  puts MXNet::NDArray.random_uniform(shape: 1).as_scalar
end

10.times do |i|
  puts MXNet::NDArray.random_randint(1, 100, shape: 1).as_scalar
end
