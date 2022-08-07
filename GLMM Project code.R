library(tidyverse)
library(ggplot2)
library(GGally)
library(pscl)
library(Metrics)
library(lme4)
library(glmmTMB)

# First importing the data in R

oldat <- read_csv("data/rioolympics.csv", na = "#N/A")
oldat <- oldat %>% mutate(
  soviet = as_factor(soviet),
  comm = as_factor(comm),
  muslim = as_factor(muslim),
  oneparty = as_factor(oneparty),
  host = as_factor(host))

oldat_transformed <- oldat %>%
  gather(key, value, -country, -country.code, -soviet, -comm, -muslim, -oneparty, -altitude, -host) %>%
  extract(key, c('key', 'year'), '([[:alpha:]]+)([[:digit:]]+)')

# Then transforming the data so that each year for each country would be a an observation in the seperate row
oldat_transformed <- oldat %>% 
  gather(key, value, -country, -country.code, -soviet, -comm, -muslim, -oneparty, -altitude, -host) %>% 
  extract(key, c("key", "year"), "([[:alpha:]]+)([[:digit:]]+)") %>%
  spread(key, value) %>%  
  mutate(year = as_factor(str_c("20", year))) %>% 
  mutate(gdp = gdp/1000000,
         pop = pop/1000000)

# And split the data into training data based on which we will build the model and test data, on which we will check accuracy of predition.
# Extracting test data (medals in 2016)
oldat_test <- oldat_transformed %>% 
  filter(year == "2016") %>% 
  na.omit()   # We removed test data with missing observations in order to properly calculate RMSE.


# And training data (all other years)
oldat_training <- oldat_transformed %>% 
  filter(year != "2016") %>% na.omit()


# Fitting models ----------------------------------------------------------
# Poisson GLMM
poisson.glmm.preliminary <- glmer(tot ~ athletes + gdp + pop + host + oneparty + muslim + soviet + comm + altitude + (1|country), 
                               family = poisson,
                               data = oldat_training,
                               control = glmerControl(check.nobs.vs.nRE = "ignore"))  # the ovvriding of check of number of observations against
                                                                                      # is added in order to allow to allow for the model with 1
                                                                                      # with 1 missing observation (GPD is NA for 2002 for Afganistan)
summary(poisson.glmm.preliminary)
# Looking at the p-value, only athletes, host and comm are considered variables with of statistical significance so we will use these in the model 
poisson.glmm.final <- glmer(tot ~ athletes + host + comm + (1|country),
                            family = poisson,
                            data = oldat_training)
summary(poisson.glmm.final)
confint(poisson.glmm.final)
poisson.glmm.final.fit <- round(predict(poisson.glmm.final, newdata = oldat_test, type = "response"),0)
poisson.glmm.final.RMSE <- round(rmse(oldat_test$tot,poisson.glmm.final.fit), digits = 3)
poisson.glmm.final.RMSE


# Zero inflated poisson mixed model
zeroinfl.glmm.preliminary <- glmmTMB(tot ~ athletes + gdp + pop + host + oneparty + muslim + soviet + comm + altitude + (1|country), family = poisson, data = oldat_test)
summary(zeroinfl.glmm.preliminary)
confint(zeroinfl.glmm.preliminary) # double check of significant variables with calculation of confidence intervals

zeroinfl.glmm.final <- glmmTMB(tot ~ athletes + comm + (1|country), family = poisson, data = oldat_test)
summary(zeroinfl.glmm.final)
confint(zeroinfl.glmm.final)
zeroinfl.glmm.final.fit <- round(predict(zeroinfl.glmm.final, newdata = oldat_test, type = "response"),0)
zeroinfl.glmm.final.RMSE <- round(rmse(oldat_test$tot,zeroinfl.glmm.final.fit), digits = 3)
zeroinfl.glmm.final.RMSE

# Linear mixed model
lmm.preliminary <- lmer(tot ~ athletes + gdp + pop + host + oneparty + muslim + soviet + comm + altitude + (1|country),
                        data = oldat_training,
                        control = lmerControl(check.nobs.vs.nRE = "ignore"))  # the ovvriding of check of number of observations against
summary(lmm.preliminary)
confint(lmm.preliminary)

# Linear mixed model does not have p-values so we will estimate confidence intevals
# They show that athletes, gdp, pop and oneparty are statistically significant predictors in the model as all other intervals contain 0.
# Also it should be mentioned that all significant predictors have coefficient above 0 that means that the correlation is positive with number of medals.
lmm.final <- lmer(tot ~ athletes + gdp + pop + oneparty + (1|country),
                  data = oldat_training)
summary(lmm.final)
confint(lmm.final)
lmm.final.fit <- round(predict(lmm.final, newdata = oldat_test, type = "response"),0)
lmm.final.RMSE <- round(rmse(oldat_test$tot,lmm.final.fit), digits = 3)
lmm.final.RMSE

# Poisson model (from Assignment A for comparison)
poisson.glm.final <- glm(tot ~ athletes + gdp + pop + host + oneparty + muslim + comm, 
                         family = poisson(),
                         data = oldat_training)
summary(poisson.glm.final)

poisson.glm.final.fit <- round(predict(poisson.glm.final, newdata = oldat_test, type = "response"), digits = 0)
poisson.glm.final.RMSE <- round(rmse(oldat_test$tot,poisson.glm.final.fit), digits = 3)
poisson.glm.final.RMSE

# Zeroinflated model (from Assignment A for comparison)
zeroinfl.model <- zeroinfl(tot ~ gdp + athletes + pop + host + oneparty + muslim + comm, data = oldat_training)
summary(zeroinfl.model)
zeroinfl.model.fit <- round(predict(zeroinfl.model, newdata = oldat_test, type = "response"),0)
zeroinfl.model.RMSE <- round(rmse(oldat_test$tot, zeroinfl.model.fit), digits = 3)
zeroinfl.model.RMSE

# Linear regression model (from Assignment A for comparison)
gaussian.glm.final <- glm(tot ~ athletes + pop + gdp + oneparty + soviet, 
                          family = gaussian,
                          data = oldat_training)

summary(gaussian.glm.final)
gaussian.glm.final.fit <- round(predict(gaussian.glm.final, newdata = oldat_test, type = "response"), 0)
gaussian.glm.final.RMSE <- round(rmse(oldat_test$tot, gaussian.glm.final.fit), digits = 3)
gaussian.glm.final.RMSE


# Summary of comparison of selected GLMs and GLMMs: 
fm <- list("Linear" = gaussian.glm.final, "Poison" = poisson.glm.final, "Zero Inflated" = zeroinfl.model,
           "Linear Mixed Model" = lmm.final, "Poisson GLMM" = poisson.glmm.final, "Zero Inflated GLMM" = zeroinfl.glmm.final)
rbind(AIC = sapply(fm, function(x) round(AIC(x),0)),
      RSME = c(gaussian.glm.final.RMSE, poisson.glm.final.RMSE, zeroinfl.model.RMSE,
               lmm.final.RMSE, poisson.glmm.final.RMSE, zeroinfl.glmm.final.RMSE))

# Preparing table for visualizing actual vs predicted results for GLMs and GLMMs
set.seed(6)
plot_table <- oldat_test %>% select(country, tot) %>% 
  mutate(actual = tot,
         predict_poisson_GLMM = poisson.glmm.final.fit,
         predict_zeroinfl_GLMM = zeroinfl.glmm.final.fit,
         predict_LMM = lmm.final.fit,
         predict_poisson_GLM = poisson.glm.final.fit,
         predict_zeroinfl_GLM = zeroinfl.model.fit,
         predict_LM = gaussian.glm.final.fit) %>% 
  select(-tot) %>% 
  sample_n(size = 15)
plot_table

# Creating the plot to visualize actual vs predicted results for GLMs and GLMMs
colors <- c("Actual number of madels in 2016" = "#030200",
            "Predicted by Poisson GLMM" = "#ff2d26", 
            "Predicted by Poisson GLM" = "#26ff4a", 
            "Predicted by Zeroinflated GLMM" = "#e874ff", 
            "Predicted by Zeroinflated GLM" = "#fff674",
            "Predicted by Linear mixed model" = "#291eff", 
            "Predicted by Linear model" = "#ff9a1e") #will be used in legend to plots

plot_table %>% ggplot(aes(x = country)) +
  geom_jitter(aes(y = actual, color = "Actual number of madels in 2016" ), size = 6, width = 0, height = 0) + 
  geom_jitter(aes(y = predict_poisson_GLMM, color = "Predicted by Poisson GLMM" ), size = 4, width = 0.2, height = 0, alpha = 0.8) + 
  geom_jitter(aes(y = predict_poisson_GLM, color = "Predicted by Poisson GLM" ), size = 4, width = 0.2, height = 0, alpha = 0.8) + 
  geom_jitter(aes(y = predict_zeroinfl_GLMM, color = "Predicted by Zeroinflated GLMM" ), size = 4, width = 0.2, height = 0, alpha = 0.8) + 
  geom_jitter(aes(y = predict_zeroinfl_GLM, color = "Predicted by Zeroinflated GLM" ), size = 4, width = 0.2, height = 0, alpha = 0.8) + 
  geom_jitter(aes(y = predict_LMM, color = "Predicted by Linear mixed model" ), size = 4, width = 0.2, height = 0, alpha = 0.8) + 
  geom_jitter(aes(y = predict_LM, color = "Predicted by Linear model" ), size = 4, width = 0.2, height = 0, alpha = 0.8) + 
  ggtitle("Comparison of prediction of GLMs and GLMMs on example of 15 random countries") + 
  labs(x = "Country",
       y = "Number of medals",
       color = "Legend",
       fill = "") +
  scale_color_manual(values = colors) +
  scale_y_continuous(breaks = round(seq(min(plot_table[,-1]), max(plot_table[,-1]), by = 3))) +
  theme(legend.position="bottom")

