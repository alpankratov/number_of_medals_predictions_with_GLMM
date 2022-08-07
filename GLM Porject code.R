library(tidyverse)
library(ggplot2)
library(GGally)
library(pscl)
library(Metrics)


# First importing the data in R
oldat <- read_csv("data/rioolympics.csv", na = "#N/A")
oldat <- oldat %>% mutate(
  soviet = as_factor(soviet),
  comm = as_factor(comm),
  muslim = as_factor(muslim),
  oneparty = as_factor(oneparty),
  host = as_factor(host))


# Then transforming the data so that each year for each country would be a an observation in the seperate row 
oldat_transformed <- oldat %>% 
  gather(key, value, -country, -country.code, -soviet, -comm, -muslim, -oneparty, -altitude, -host) %>% 
  extract(key, c("key", "year"), "([[:alpha:]]+)([[:digit:]]+)") %>%
  spread(key, value) %>%  
  mutate(year = as_factor(str_c("20", year)))

# And split the data into training data based on which we will build the model and test data, on which we will check accuracy of predition.
# Extracting test data (medals in 2016)
oldat_test <- oldat_transformed %>% 
  filter(year == "2016") %>% 
  na.omit()   # We removed test data with missing observations in order to properly calculate RMSE.


# And training data (all other years)
oldat_training <- oldat_transformed %>% 
  filter(year != "2016")


# Now we will explore  relationship between variables. Numerical first.
oldat_training %>% 
  dplyr::select(-country, -country.code, -totgold, -totmedals, -year, -host, -soviet, -comm, -muslim, -oneparty) %>% 
  ggpairs(lower = list(continuous = wrap("points", alpha = 0.3, color = "#186f7a")),
          diag = list(continuous = wrap("barDiag", colour = "#186f7a")),
          upper = list(continuous = wrap("cor", size = 5)))
# There is a very strong positive correlation between total number of medals won and the number of gold medals won (0.97).
# Also other variable correlate with toal and gold medal won in a similar way, therefore we will focus only on prediction of total number of medals won.
# And disregard gold medals won as prediction of total number of all medals won in the Olympics will be also a good prediction for the number of gold medals.

# There is a strong positive correlation between number of medal won (both total and gold) and the number of athletes representing the country. (corr. 0.887)
# Also GDP is a strong indicator of number of medals the country wins in the Olympics (corr. coefficient is 0.76) 

# Population is not as strongly correlated with medals won as GPD and number of athelets (corr. coefficient 0.41) 

# It looks like altitude has almost no correlation with any other numerical variable (corr. coefficient is approx -0.1 with all variables)


# Now let's explore relationshp between variables (categorical)
oldat_training %>% 
  dplyr::select(-country, -country.code, -totgold, -totmedals, -year, -gdp, -athletes, -altitude, -pop) %>%
  ggpairs(lower = list(discrete = wrap("facetbar", fill = "#186f7a"),
                       combo = wrap("facethist", color = "#186f7a")),
          diag = list(discrete = wrap("barDiag", fill = "#186f7a")),
          upper = list(discrete = wrap("ratio", fill = "#186f7a")))
# From the plot it looks like that ex-soviet and communist countries have slightly more medal on average than non-communist
# and muslim countries have lower number of medals won than non muslim, but if the difference significant is not clear at this stage.
# On the other hand, countries that hosted Olympics tend to have larger number of medals won then those that did not host.
# Also countries with one-party system tend to win more medals on average. But this probably due to China as there are only three countries 
# with one-party system.
oldat_training %>% filter(oneparty == 1) 


# After exploring the data, let's build genral linear models. First we will try to build poisson regression model using all variables (preliminary).
# As we are building model to predict number of medal, i.e. counts, not rates, we will not use offset in the model.
# However, we also could have predicted number of medals won per athlete or per 1 person and in this case we would have used
# expected number of medals won allowed for differences in population or number of athletes presenting the country as offset in the model.
# But here we will have population and athletes directly as predictors

poisson.glm.preliminary <- glm(tot ~ athletes + gdp + pop + host + oneparty + muslim + soviet + comm + altitude, 
                                 family = poisson,
                                 data = oldat_training)
summary(poisson.glm.preliminary)


drop1(poisson.glm.preliminary, test = "F")

# Looking at the summary of the poisson regression model (Wald test), "soviet" and "altitude" coefficients in the above model has a p-value above 0.05
# F-test provides the same results. It means that these variables are not significant as expected from the exploratory analysis above and therefore can be removed from the model
# Now let's make a new model without these variables:

poisson.glm.final <- glm(tot ~ athletes + gdp + pop + host + oneparty + muslim + comm, 
                   family = poisson(),
                   data = oldat_training)
summary(poisson.glm.final)

poisson.glm.final.fit <- predict(poisson.glm.final, newdata = oldat_test, type = "response")
poisson.glm.final.RMSE <- round(rmse(oldat_test$tot,poisson.glm.final.fit), digits = 3)
poisson.glm.final.RMSE
poisson.glm.final.RMSE == round(sqrt(sum((oldat_test$tot-poisson.glm.final.fit)^2)/length(oldat_test$tot)), digits = 3) # to check rmse() function

# Although all variables in this model are considered significant (t-value is below 0.05)
# Looking at the residual deviance (1477), it is significantly higher that amount of chi squared 
# on 423 degrees of freedom and 95% significance level (472) indicating a poor fit if the Poisson is the correct model for the response.
qchisq(df = 423, p = 0.95)

# It is likely due to overdispersion. Below is the plot of dispersion of poisson distribution. 
# We can see in the plots that there are residuals for counts close to 0 that are above +8 of standard Pearson distribution.
poisson.dispersion  <- tibble(pred.poisson = predict(poisson.glm.final, type = "response"),
                              stand.resid.poisson = rstandard(model = poisson.glm.final, type = "pearson"))

poisson.dispersion %>% 
  ggplot(aes(x = pred.poisson, y = stand.resid.poisson)) +
  geom_point() +
  labs(x = "Predicted count", y = "Standardised Pearson residuals")

# Therefore, overdispersion is probably observed due to excess zeros in the data
# as lot of countries has not won a signle medal (27.5% of total training data):
sum(oldat_training$tot == 0)/length(oldat_training$tot)
sum(oldat_test$tot == 0)/length(oldat_test$tot)

# Excess of zeroes is caused by the fact that the number of medals is finite and some countries end up with 0 medals
# For this particular case, zero-inflated models may be more appropriate. We will use the same variables for zero inflated model as for poisson

zeroinfl.model <- zeroinfl(tot ~ gdp + athletes + pop + host + oneparty + muslim + comm, data = oldat_training)
summary(zeroinfl.model)

zeroinfl.model.fit <- predict(zeroinfl.model, newdata = oldat_test, type = "response")
zeroinfl.model.RMSE <- round(rmse(oldat_test$tot, zeroinfl.model.fit), digits = 3)
zeroinfl.model.RMSE


# We can check if zero inflated model better fits the data then Poisson model by performing a Vuong test of the two models.
# The Vuong non-nested test is based on a comparison of the predicted probabilities of two models that do not nest. 
# Examples include comparisons of zero-inflated count models with their non-zero-inflated analogs 
# (e.g., zero-inflated Poisson versus ordinary Poisson, or zero-inflated negative-binomial versus ordinary negative-binomial). 
# A large, positive test statistic provides evidence of the superiority of model 1 over model 2, while a large, 
# negative test statistic is evidence of the superiority of model 2 over model 1. Under the null that the models are indistinguishable, 
# the test statistic is asymptotically distributed standard normal.
vuong(zeroinfl.model, poisson.glm.final)
# The Vuong test compares the zero-inflated model with an ordinary Poisson regression model. 
# Here we can see that our test statistic is significant, indicating that the zero-inflated model (model 1) is superior 
# to the standard Poisson model (model 2).


# Finaly, for the sake of comparison we will also build basic type of linear model based on normal distribution with identity log-link 
# (i.e. the sale as lm() finction in R)
# We will start with the model that includes all predictors
gaussian.glm.preliminary <- glm(tot ~ athletes + gdp + pop + host + oneparty + muslim + soviet + comm + altitude, 
                                family = gaussian,
                                data = oldat_training)

summary(gaussian.glm.preliminary)


# t-values for coefficients of Muslim, host comm and altitude variables are significantly higher than 0.05 therefore they 
# are not statistically significant so we remove it from the final version of the basic linear model.  
gaussian.glm.final <- glm(tot ~ athletes + pop + gdp + oneparty + soviet, 
                          family = gaussian,
                          data = oldat_training)

summary(gaussian.glm.final)
gaussian.glm.final.fit <- predict(gaussian.glm.final, newdata = oldat_test, type = "response")
sum(gaussian.glm.final.fit)
gaussian.glm.final.RMSE <- round(rmse(oldat_test$tot, gaussian.glm.final.fit), digits = 3)
gaussian.glm.final.RMSE


# And finally let's prepare a summary of comparison of 3 selected models: 
fm <- list("Linear" = gaussian.glm.final, "Poison" = poisson.glm.final, "Zero Inflated" = zeroinfl.model)
rbind(AIC = sapply(fm, function(x) AIC(x)),
      Deviance = sapply(fm, function(x) deviance(x)),
      Degrees_of_freedom = sapply(fm, function(x) df.residual(x)),
      Chi_squared = qchisq(sapply(fm, function(x) df.residual(x)), p = 0.95),
      RSME = c(gaussian.glm.final.RMSE, poisson.glm.final.RMSE, zeroinfl.model.RMSE))
  
# To remove "atheletes" variable from each selected model.
# Poisson model
poisson.glm.final.wo.athletes <- glm(tot ~ gdp + pop + host + oneparty + muslim + comm, family = poisson(), data = oldat_training)
summary(poisson.glm.final.wo.athletes)
poisson.glm.final.fit.wo.athletes <- predict(poisson.glm.final.wo.athletes, newdata = oldat_test, type = "response")
poisson.glm.final.RMSE.wo.athletes <- round(rmse(oldat_test$tot,poisson.glm.final.fit.wo.athletes), digits = 3)
poisson.glm.final.RMSE.wo.athletes

# Zero inflated model
zeroinfl.model.wo.athletes <- zeroinfl(tot ~ gdp + pop + host + oneparty + muslim + comm, data = oldat_training)
summary(zeroinfl.model.wo.athletes)
zeroinfl.model.fit.wo.athletes <- predict(zeroinfl.model.wo.athletes, newdata = oldat_test, type = "response")
zeroinfl.model.RMSE.wo.athletes <- round(rmse(oldat_test$tot, zeroinfl.model.fit.wo.athletes), digits = 3)
zeroinfl.model.RMSE.wo.athletes

# Gaussinal model
gaussian.glm.final.wo.athletes <- glm(tot ~ gdp + pop + oneparty + soviet, family = gaussian, data = oldat_training)
summary(gaussian.glm.final.wo.athletes)
gaussian.glm.final.fit.wo.athletes <- predict(gaussian.glm.final.wo.athletes, newdata = oldat_test, type = "response")
gaussian.glm.final.RMSE.wo.athletes <- round(rmse(oldat_test$tot, gaussian.glm.final.fit.wo.athletes), digits = 3)
gaussian.glm.final.RMSE.wo.athletes

# Check summary of good fit criateria
fm.wo.athletes <- list("Linear" = gaussian.glm.final.wo.athletes, "Poison" = poisson.glm.final.wo.athletes, "Zero Inflated" = zeroinfl.model.wo.athletes)
rbind(AIC = sapply(fm.wo.athletes, function(x) AIC(x)),
      Deviance = sapply(fm.wo.athletes, function(x) deviance(x)),
      Degrees_of_freedom = sapply(fm.wo.athletes, function(x) df.residual(x)),
      Chi_squared = qchisq(sapply(fm.wo.athletes, function(x) df.residual(x)), p = 0.95),
      RSME = c(gaussian.glm.final.RMSE.wo.athletes, poisson.glm.final.RMSE.wo.athletes, zeroinfl.model.RMSE.wo.athletes))


# And finally we will prepare models that include only 'athletes' variable
# Poisson model

poisson.glm.final.only.athletes <- glm(tot ~ athletes, family = poisson(), data = oldat_training)
summary(poisson.glm.final.only.athletes)
poisson.glm.final.fit.only.athletes <- predict(poisson.glm.final.only.athletes, newdata = oldat_test, type = "response")
poisson.glm.final.RMSE.only.athletes <- round(rmse(oldat_test$tot,poisson.glm.final.fit.only.athletes), digits = 3)
poisson.glm.final.RMSE.only.athletes

# Zero inflated model
zeroinfl.model.only.athletes <- zeroinfl(tot ~ athletes, data = oldat_training)
summary(zeroinfl.model.only.athletes)
zeroinfl.model.fit.only.athletes <- predict(zeroinfl.model.only.athletes, newdata = oldat_test, type = "response")
zeroinfl.model.RMSE.only.athletes <- round(rmse(oldat_test$tot, zeroinfl.model.fit.only.athletes), digits = 3)
zeroinfl.model.RMSE.only.athletes

# Gaussinal model
gaussian.glm.final.only.athletes <- glm(tot ~ athletes, family = gaussian, data = oldat_training)
summary(gaussian.glm.final.only.athletes)
gaussian.glm.final.fit.only.athletes <- predict(gaussian.glm.final.only.athletes, newdata = oldat_test, type = "response")
gaussian.glm.final.RMSE.only.athletes <- round(rmse(oldat_test$tot, gaussian.glm.final.fit.only.athletes), digits = 3)
gaussian.glm.final.RMSE.only.athletes

# Check summary of good fit criateria
fm.only.athletes <- list("Linear" = gaussian.glm.final.only.athletes, "Poison" = poisson.glm.final.only.athletes, "Zero Inflated" = zeroinfl.model.only.athletes)
rbind(AIC = sapply(fm.only.athletes, function(x) AIC(x)),
      Deviance = sapply(fm.only.athletes, function(x) deviance(x)),
      Degrees_of_freedom = sapply(fm.only.athletes, function(x) df.residual(x)),
      Chi_squared = qchisq(sapply(fm.only.athletes, function(x) df.residual(x)), p = 0.95),
      RSME = c(gaussian.glm.final.RMSE.only.athletes, poisson.glm.final.RMSE.only.athletes, zeroinfl.model.RMSE.only.athletes))

##### Appendix 1. Comparison of models with pop and gdp divided by 1000
oldat_transformed2 <- oldat %>% 
  gather(key, value, -country, -country.code, -soviet, -comm, -muslim, -oneparty, -altitude, -host) %>% 
  extract(key, c("key", "year"), "([[:alpha:]]+)([[:digit:]]+)") %>%
  spread(key, value) %>%  
  mutate(year = as_factor(str_c("20", year)),
         gdp = gdp/1000000,
         pop = pop/1000000)

oldat_test2 <- oldat_transformed2 %>% 
  filter(year == "2016") %>% 
  na.omit()   # We removed test data with missing observations in order to properly calculate RMSE.


# And training data (all other years)
oldat_training2 <- oldat_transformed2 %>% 
  filter(year != "2016")

oldat_training2

zeroinfl.model2 <- zeroinfl(tot ~ gdp + athletes + pop + host + oneparty + muslim + comm, data = oldat_training2)
summary(zeroinfl.model2)


zeroinfl.model.fit2 <- predict(zeroinfl.model2, newdata = oldat_test2, type = "response")
zeroinfl.model.RMSE2 <- round(rmse(oldat_test2$tot, zeroinfl.model.fit2), digits = 3)
zeroinfl.model.RMSE2

#comparison both zero inflated models
comp <- list("Zero Inflated v.1" = zeroinfl.model, "Zero Inflated v.2" = zeroinfl.model2)
rbind(AIC = sapply(comp, function(x) AIC(x)),
      Deviance = sapply(comp, function(x) deviance(x)),
      RSME = c(zeroinfl.model.RMSE, zeroinfl.model.RMSE2))
# The characteristics are the same