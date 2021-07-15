require "mxnet"
require "mxnet/gluon"
require "ishi/iterm2"

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

Ishi.new(width: 70) do
  scatter(x[.., 1], y)
end

w = MXNet::NDArray.random_normal(shape: [num_inputs, num_outputs], ctx: model_ctx)
b = MXNet::NDArray.random_normal(shape: [num_outputs], ctx: model_ctx)
params = [w, b]

params.each(&.attach_grad)

def net(x, w, b)
  x.dot(w) + b
end

def square_loss(yhat, y)
  ((yhat - y) ** 2).mean
end

def sgd(params, lr)
  params.each do |param|
    if grad = param.grad
      param[..] = param - lr * grad
    end
  end
end

losses = [] of Float64

def plot(losses, x, w, b)
  figure = Ishi.new
  figure.canvas_size(1280, 480)
  charts = figure.charts(1, 2)
  charts[0].plot(losses, title: "Loss")
  charts[1].plot(x[..100, 1], net(x[..100, ..], w, b)[.., 0], "or", title: "Estimated")
  charts[1].plot(x[..100, 1], real_fn(x[..100, ..]), "+g", title: "Real")
  figure.show(width: 140)
end

plot(losses, x, w, b)

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
  losses << cumulative_loss / num_batches
  puts cumulative_loss / num_batches
end

plot(losses, x, w, b)

params.each do |param|
  puts param
end
