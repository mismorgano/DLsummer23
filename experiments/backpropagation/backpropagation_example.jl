import Random
import LinearAlgebra
import Plots

function σ(z)
    1 / (1 + exp(-z))
end

function ReLU(z)
    max(0, z)
end
function activate(x, W, b, activation)
    activation.(W * x + b)
end

function cost(W2, W3, W4, b2, b3, b4, activation)
    costvec = zeros(10)
    for i = 1:10
        x = [x1[i]; x2[i]]
        a2 = activate(x, W2, b2, activation)
        a3 = activate(a2, W3, b3, activation)
        a4 = activate(a3, W4, b4, activation)
        costvec[i] = LinearAlgebra.norm(y[:, i] - a4)
    end
    LinearAlgebra.norm(costvec)^2
end

function training(activation)
    # Initialize weights and biases
    Random.seed!(1000)
    W2 = rand(2, 2) .* 0.5
    W3 = rand(3, 2) .* 0.5
    W4 = rand(2, 3) .* 0.5
    b2 = rand(2) .* 0.5
    b3 = rand(3) .* 0.5
    b4 = rand(2) .* 0.5


    # Forward and Back propagate
    η = 0.05
    Niter = Int64(1e6)
    savecost = zeros(Niter)
    for counter = 1:Niter
        k = rand(1:10)
        x = [x1[k]; x2[k]]
        # Forward pass
        a2 = activate(x, W2, b2, activation)
        a3 = activate(a2, W3, b3, activation)
        a4 = activate(a3, W4, b4, activation)
        # Backward pass
        delta4 = @. a4 * (1 - a4) * (a4 - y[:, k])
        delta3 = (@. a3 * (1 - a3)) .* (transpose(W4) * delta4)
        delta2 = (@. a2 * (1 - a2)) .* (transpose(W3) * delta3)
        # Gradient step
        W2 -= η * delta2 * transpose(x)
        W3 -= η * delta3 * transpose(a2)
        W4 -= η * delta4 * transpose(a3)
        b2 -= η * delta2
        b3 -= η * delta3
        b4 -= η * delta4
        # Monitor progress
        newcost = cost(W2, W3, W4, b2, b3, b4, activation)
        savecost[counter] = newcost
    end
    # Plot Classifier
    # PlotClassifier(activation)
    #Plot progress
    # Plots.plot(savecost)
    W2, W3, W4, b2, b3, b4, savecost
end
# Data
x1 = [0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7]
x2 = [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]
y = [ones(1, 5) zeros(1, 5); zeros(1, 5) ones(1, 5)]


function nn(x, W2, W3, W4, b2, b3, b4, activation)
    a2 = activate(x, W2, b2, activation)
    a3 = activate(a2, W3, b3, activation)
    a4 = activate(a3, W4, b4, activation)
    a4
end

function classifier(x, y, W2, W3, W4, b2, b3, b4, activation)
    (a, b) = nn([x; y], W2, W3, W4, b2, b3, b4, activation)
    if a > b
        return "aliceblue" # class [1, 0]
    else
        return "plum" # class [0, 1]
    end
end

## Set limits and remove legend
Plots.plot!(xlims=(0, 1), ylims=(0, 1))
Plots.plot!(legend=false)

function PlotClassifier(W2, W3, W4, b2, b3, b4, activation)
    ## Plot region
    a = range(0, 1, 100)

    b = [(x, y) for x in a, y in a]

    colors = [classifier(x, y, W2, W3, W4, b2, b3, b4, activation) for (x, y) in b]

    plot = Plots.scatter(first.(b), last.(b), color=colors)

    ## Plot blue points [1, 0]
    Plots.scatter!(plot, x1[1:5], x2[1:5], color="blue")

    ## Plot red points [0, 1]
    Plots.scatter!(plot, x1[6:10], x2[6:10], color="red")
    return plot
end
