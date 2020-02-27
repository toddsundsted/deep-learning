require "mxnet"

MXNet::Random.seed(1)

x = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
x.attach_grad

MXNet::Autograd.record do
  y = x * 2
  z = y * x
  z.backward
end

puts x.grad

a = MXNet::NDArray.random_normal(shape: 3)
a.attach_grad

MXNet::Autograd.record do
  b = a * 2.0
  while (b.norm < 1000.0).as_scalar > 0.0
    b = b * 2.0
  end
  if (b.sum > 0.0).as_scalar > 0.0
    c = b
  else
    c = 100.0 * b
  end
  c.backward
end

puts a.grad
