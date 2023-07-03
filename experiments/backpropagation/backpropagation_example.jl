import Random
import LinearAlgebra
function σ(z)
    1/(1+exp(-z))
end

function activate(x, W, b)
    σ.(W*x + b)
end

function cost(W2, W3, W4, b2, b3, b4)
    costvec = zeros(10)
    for i = 1:10
        x = [x1[i]; x2[i]]
        a2 = activate(x, W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        costvec[i] = LinearAlgebra.norm(y[:, i] - a4)
    end
    LinearAlgebra.norm(costvec)^2 
end

# Data
x1 = [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7]
x2 = [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]
y =  [ones(1, 5) zeros(1, 5); zeros(1, 5) ones(1, 5)]

# Initialize weights and biases
Random.seed!(100)
W2 = rand(2, 2).*0.5
W3 = rand(3, 2).*0.5
W4 = rand(2, 3).*0.5
b2 = rand(2).*0.5
b3 = rand(3).*0.5
b4 = rand(2).*0.5

# Forward and Back propagate
eta = 0.5
Niter = Int64(1e2)
savecost = zeros(Niter)
for counter = 1:Niter
    k = rand(1:10)
    x = [x1[k]; x2[k]]
    # Forward pass
    a2 = activate(x, W2, b2)
    a3 = activate(a2, W3, b3)
    a4 = activate(a3, W4, b4)
    # Backward pass
    delta4 = @. a4 * (1-a4) * (a4 - y[:, k])
    delta3 = (@. a3 * (1-a3)) .* (transpose(W4) * delta4)
    delta2 = (@. a2 * (1-a2)) .* (transpose(W3) * delta3)
    # Gradient step
    global W2 -= eta * delta2*transpose(x)
    global W3 -= eta*delta3*transpose(a2)
    global W4 -= eta*delta4*transpose(a3)
    global b2 -= eta*delta2
    global b3 -= eta*delta3
    global b4 -= eta*delta4
    # Monitor progress
    newcost = cost(W2, W3, W4, b2, b3, b4)
    savecost[counter] = newcost
end
