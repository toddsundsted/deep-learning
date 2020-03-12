require "mxnet/gluon"
require "ishi/iterm2"

MXNet::Random.seed(1)

def transform(data, label)
  {(data / 128).floor, label.to_f32}
end

mnist_train = MXNet::Gluon::Data::Vision::MNIST.new(train: true, transform: ->transform(MXNet::NDArray, Int32))
mnist_test = MXNet::Gluon::Data::Vision::MNIST.new(train: false, transform: ->transform(MXNet::NDArray, Int32))

ycount = MXNet::NDArray.ones([10])
xcount = MXNet::NDArray.ones([784, 10])

mnist_train.each do |data, label|
  x = data.reshape([784])
  y = label.to_i32
  xcount[.., y] += x
  ycount[y] += 1
end

10.times do |i|
  xcount[.., i] = xcount[.., i] / ycount[i]
end

Ishi.new(width: 70) do
  canvas_size(28 * 10 + 10, 28 + 10)
  margin(0.2, 0.2, 0.2, 0.2)
  palette(:hot)
  show_colorbox(false)
  show_border(false)
  show_xtics(false)
  show_ytics(false)
  show_key(false)
  charts(1, 10) do |i|
    values = xcount[.., i].reshape([28, 28])
    imshow(values)
  end
end

py = ycount / ycount.sum
puts py

logxcount = xcount.log
logxcountneg = (1 - xcount).log
logpy = py.log

figure = Ishi.new
figure.canvas_size(2000, 150)
figure.show_ytics(false)
figure.show_key(false)
charts = figure.charts(1, 10)

ctr = 0
mnist_test.each do |data, label|
  x = data.reshape([784])
  y = label.to_i32

  break if ctr == 10

  logpx = logpy.copy_to(logpy.context)
  10.times do |i|
    logpx[i] += logxcount[.., i].dot(x) + logxcountneg[.., i].dot(1 - x)
  end
  logpx -= logpx.max

  px = logpx.exp
  px /= px.sum

  charts[ctr].plot([label], [1.0], style: :boxes, lw: 2)
  charts[ctr].plot(px, lw: 2)

  ctr += 1
end

figure.show(width: 200)
