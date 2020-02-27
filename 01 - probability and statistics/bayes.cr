require "mxnet/gluon"

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

require "iterm2"
require "stumpy_png"

include StumpyPNG

canvas = Canvas.new(28 * 10, 28)

10.times do |i|
  28.times do |x|
    28.times do |y|
      value = xcount[.., i].reshape([28, 28])[y, x].as_scalar * 256
      color = RGBA.from_rgb_n(value, value, value, 8)
      canvas[i * 28 + x, y] = color
    end
  end
end

Iterm2.new.display(width: 70) do |io|
  StumpyPNG.write(canvas, io)
end

py = ycount / ycount.sum
puts py

logxcount = xcount.log
logxcountneg = (1 - xcount).log
logpy = py.log

require "ishi/iterm2"

ctr = 0
mnist_test.each do |data, label|
  x = data.reshape([784])
  y = label.to_i32

  break if ctr == 10
  ctr += 1

  logpx = logpy.copy_to(logpy.context)
  10.times do |i|
    logpx[i] += logxcount[.., i].dot(x) + logxcountneg[.., i].dot(1 - x)
  end
  logpx -= logpx.max

  px = logpx.exp
  px /= px.sum

  Ishi.new do
    plot(px.to_a(Float32))
    plot([label.as(Float32)], [1.0], style: :boxes)
  end
end
