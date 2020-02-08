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
  2 * x[.., 0] - 3.4 * x[.., 1] + 4.2
end

x = MXNet::NDArray.random_normal(shape: [num_examples, num_inputs], ctx: data_ctx)
noise = 0.1 * MXNet::NDArray.random_normal(shape: [num_examples], ctx: data_ctx)
y = real_fn(x) + noise

net = MXNet::Gluon::NN::Dense.new(1)
net.collect_params.init(ctx: model_ctx)

square_loss = MXNet::Gluon::Loss::L2Loss.new

trainer = MXNet::Gluon::Trainer.new(net.collect_params, :sgd, lr: learning_rate)

train_data = MXNet::Gluon::Data::DataLoader(Tuple(MXNet::NDArray, MXNet::NDArray), Tuple(MXNet::NDArray, MXNet::NDArray)).new(
  MXNet::Gluon::Data::ArrayDataset.new(x, y),
  batch_size: batch_size, shuffle: true
)

epochs.times do |epoch|
  cumulative_loss = 0
  train_data.rewind.each do |(data, label)|
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    loss =
      MXNet::Autograd.record do
        output = net.call([data]).first
        square_loss.call([output, label]).first
      end
    loss.backward
    trainer.step(batch_size)
    cumulative_loss += MXNet::NDArray.mean(loss).as_scalar
  end
  puts cumulative_loss / num_batches
end

net.collect_params.each do |_, param|
  puts param.data
end
