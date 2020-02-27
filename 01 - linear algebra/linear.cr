require "mxnet"

# Scalars

x = MXNet::NDArray.array([3.0])
y = MXNet::NDArray.array([2.0])

puts x + y

puts x * y

puts x / y

puts MXNet::NDArray.power(x, y)

puts x.as_scalar

# Vectors

u = MXNet::NDArray.array([0.0, 1.0, 2.0, 3.0])
puts u

puts u[3]

# Length, dimensionality, and, shape

puts u.shape

# Matrices

a = MXNet::NDArray.zeros([5, 4])
puts a

x = MXNet::NDArray.array(0...20)
a = x.reshape([5, 4])
puts a

puts a[2, 3]

puts a[2, ..]
puts a[.., 3]

puts a.transpose

# Tensors

x = MXNet::NDArray.array(0...24).reshape([2, 3, 4])
puts x.shape
puts x

# Element-wise operations

u = MXNet::NDArray.array([1.0, 2.0, 4.0, 8.0])
v = MXNet::NDArray.ones_like(u) * 2
puts v
puts u + v
puts u - v
puts u * v
puts u / v

b = MXNet::NDArray.ones_like(a) * 3
puts b
puts a + b
puts a * b

# Basic properties of tensor arithmetic

n = 2
x = MXNet::NDArray.ones(3)
y = MXNet::NDArray.zeros(3)
puts x.shape
puts y.shape
puts (n * x).shape
puts (n * x + y).shape

# Sums and means

puts MXNet::NDArray.sum(u)

puts MXNet::NDArray.sum(a)

puts a.mean
puts a.sum / a.shape.product

# Dot products

puts MXNet::NDArray.dot(u, v)

puts MXNet::NDArray.sum(u * v)

# Matrix-vector products

a = a.as_type(:float32)
u = u.as_type(:float32)
puts MXNet::NDArray.dot(a, u)

# Matrix-matrix multiplication

a = MXNet::NDArray.ones([3, 4])
b = MXNet::NDArray.ones([4, 5])
puts MXNet::NDArray.dot(a, b)

# Norms

puts MXNet::NDArray.norm(u)

puts MXNet::NDArray.sum(MXNet::NDArray.abs(u))

puts MXNet::NDArray.norm(u, ord: 1)
