x = runif(1000, -5, 5)
y = x + rnorm(1000) + 3
xc = cbind(1, x)
sgd = function(x, y, n_iter, l_rate) {
  m = ncol(x)
  n = nrow(x)
  beta = matrix(0, nrow=m) # number of features + 1 if there is an intercept
  beta_iter = matrix(NA, nrow=n_iter, ncol=m)
  i = 0
  while (i <= n_iter) {
    sample_num = sample(n, 1)
    x_sample = x[sample_num,]
    y_sample = y[sample_num]
    derivative = x_sample %*% (x_sample%*%beta - y_sample)
    beta = beta - l_rate * derivative
    beta_iter[i,] = beta
    i = i+1 
  }
  result = list(coef=beta, coef_history=beta_iter)
  return(result)
}


beta_list = sgd(xc, y, n_iter=10000, l_rate=0.0015)
beta_list[[1]] #2 instead of 1 for beta_iter
beta_list['coef'] #coef_history for beta_iter
beta_list$coef #coef_history for beta_iter
beta = beta_list$coef
beta_history = beta_list$coef_history

model = lm(y~x)
coef = model$coefficients

beta
coef

plot(x, y, col='blue', main='linear regression with stochastic gradient descent')
for (i in c(1:5, 10, 15, seq(1000,10000, by=1000))) {
  abline(coef=beta_history[i,], col=rgb(0.8, 0, 0, 0.7))
}
abline(beta, col='yellow', lwd=10)
abline(coef, col='green', lwd=3)

xtest = runif(200, -5, 5)
ytest = xtest + rnorm(200) + 3
xtestc = cbind(1, xtest)

ytest_pred = xtestc %*% beta
mse = mean((ytest - ytest_pred)^2)

xtest_copy = data.frame(x=xtest)
ytest_pred2 = predict(model, xtest)
mse2 = mean((ytest - ytest_pred2)^2)

mse
mse2