require "mxnet"

MXNet::Random.seed(1)

# Getting started

x = MXNet::NDArray.empty([3, 4])
puts x

x = MXNet::NDArray.zeros([3, 4])
puts x

x = MXNet::NDArray.ones([3, 4])
puts x

y = MXNet::NDArray.random_normal(0, 1, shape: [3, 4])
puts y
puts y.shape

# Operations

puts x + y

puts x * y

puts x ** y

puts MXNet::NDArray.dot(x, y.transpose)

# In-place operations

MXNet::NDArray::Ops._elemwise_add(x, y, out: x)
puts x

# Slicing

puts x[1...3]

x[1, 2] = 9.0
puts x

puts x[1...2, 1...3]

x[1...2, 1...3] = 5.0
puts x

# Broadcasting

x = MXNet::NDArray.ones([3, 3])
puts x
y = MXNet::NDArray.array([0.0, 1.0, 2.0], dtype: :float32)
puts y
puts x + y

y = y.reshape(shape: [3, 1])
puts y
puts x + y
