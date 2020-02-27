require "mxnet"
require "ishi/iterm2"

MXNet::Random.seed(1)

ROLLS = 1000

probabilities = MXNet::NDArray.ones(6) / 6

rolls = MXNet::NDArray.sample_multinomial(probabilities, shape: [ROLLS])

totals = MXNet::NDArray.zeros(6)
counts = MXNet::NDArray.zeros([6, ROLLS])
rolls.each_with_index do |roll, i|
    totals[roll.as_scalar.to_i] += 1
    counts[.., i] = totals
end

puts totals / ROLLS

x = MXNet::NDArray.array(0...ROLLS, dtype: :float32).reshape([1, ROLLS]) + 1
estimates = counts / x
puts estimates[.., 0]
puts estimates[.., 1]
puts estimates[.., 10]
puts estimates[.., 100]

Ishi.new(width: 70) do
  plot(estimates[0, ..].to_a(Float32), title: "Estimated P(die=1)")
  plot(estimates[1, ..].to_a(Float32), title: "Estimated P(die=2)")
  plot(estimates[2, ..].to_a(Float32), title: "Estimated P(die=3)")
  plot(estimates[3, ..].to_a(Float32), title: "Estimated P(die=4)")
  plot(estimates[4, ..].to_a(Float32), title: "Estimated P(die=5)")
  plot(estimates[5, ..].to_a(Float32), title: "Estimated P(die=6)")
  plot("1.0/6.0")
end
