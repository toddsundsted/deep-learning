require "mxnet"
require "mxnet/gluon"

MXNet::Random.seed(1)

data_ctx = MXNet.cpu
model_ctx = MXNet.cpu

num_inputs = 2
num_outputs = 1
num_examples = 10000
batch_size = 4

epochs = 10
learning_rate = 0.0001
num_batches = num_examples / batch_size

def real_fn(x)
  2.0 * x[.., 0] - 3.4 * x[.., 1] + 4.2
end

x = MXNet::NDArray.random_normal(shape: [num_examples, num_inputs], ctx: data_ctx)
noise = 0.1 * MXNet::NDArray.random_normal(shape: [num_examples], ctx: data_ctx)
y = real_fn(x) + noise

w = MXNet::NDArray.random_normal(shape: [num_inputs, num_outputs], ctx: model_ctx)
b = MXNet::NDArray.random_normal(shape: [num_outputs], ctx: model_ctx)
params = [w, b]

params.each(&.attach_grad)

def net(x, w, b)
  MXNet::NDArray.dot(x, w) + b
end

def square_loss(yhat, y)
  MXNet::NDArray.mean((yhat - y) ** 2)
end

def sgd(params, lr)
  params.each do |param|
    if grad = param.grad
      param[..] = param - lr * grad
    end
  end
end

train_data = MXNet::Gluon::Data::DataLoader(Tuple(MXNet::NDArray, MXNet::NDArray), Tuple(MXNet::NDArray, MXNet::NDArray)).new(
  MXNet::Gluon::Data::ArrayDataset.new(x, y),
  batch_size: batch_size, shuffle: true
)

epochs.times do |epoch|
  cumulative_loss = 0.0
  train_data.rewind.each do |(data, label)|
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    loss =
      MXNet::Autograd.record do
        output = net(data, w, b)
        square_loss(output, label)
      end
    loss.backward
    sgd(params, learning_rate)
    cumulative_loss += loss.as_scalar
  end
  puts cumulative_loss / num_batches
end

params.each do |param|
  puts param
end
