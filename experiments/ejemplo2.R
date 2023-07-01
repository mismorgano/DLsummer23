
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


## En el texto se meciona que se usa minimos cuadrados
nls