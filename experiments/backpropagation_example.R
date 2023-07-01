
activate <- function(x, W, b) {
  1/(1+exp(-(W%*%x + b)))
}

cost <- function(W2, W3, W4, b2, b3, b4) {
  costvec = rep(1,10)
  for (i in 1:10) {
    x = c(x1[i], x2[i])
  }
}
## Datos 
## vistos como puntos (x1, x2)
x1 = c(0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7)
x2 = c(0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6)

## falta su clasificación pero parece que los primeors 5 son A, 
## los ultimos 5 son B

# plot(x1[1:5], x2[1:5], xlim = c(0, 1), ylim = c(0, 1))

y1 <- rbind(rep(1, 5), rep(0, 5)) # [1, 0] 5 veces
y2 <- rbind(rep(0, 5), rep(1, 5)) # [0, 1] 5 veces
y <- cbind(y1, y2) # su union

## Inicializar pesos y sesgos
W2 = 0.5*matrix(rnorm(2*2), 2)
b2 = 0.5*matrix(rnorm(2))

W3 = 0.5*matrix(rnorm(3*2), 3)
b3 = 0.5*matrix(rnorm(3))

W4 = 0.5*matrix(rnorm(2*3), 2)
b4 = 0.5*matrix(rnorm(2))

## Forward and back propagate
eta = 0.05
Niter = 1e6
savecost = rep(0, Niter)

for (counter in 1:Niter) {
  ## escogemos un punto aleatorio
  k = sample(1:10, 1)
  x = c(x1[k], x2[k])
  
  ## Forward pass
  a2 = activate(x, W2, b2)
  a3 = activate(a2, W3, b3)
  a4 = activate(a3, W4, b4)
  
  ## backward pass
  delta4 = a4*(1-a4)*(a4 - y[,k])
  delta3 = a3*(1-a3)*(t(W4) %*% delta4)
  delta2 = a2*(1-a2)*(t(W3) %*% delta3)
  
  ## gradient step
  W2 = W2 - eta*delta2%*%t(x)
  W3 = W3 - eta*delta3%*%t(a2)
  W4 = W4 - eta*delta4%*%t(a3)
  b2 = b2 - eta*delta2
  b3 = b3 - eta*delta3
  b4 = b4 - eta*delta4
  
}
