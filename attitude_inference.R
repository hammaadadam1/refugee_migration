library(tidyverse)
library(extracat)
library(rstan)
library(ggridges)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

setwd("~/Desktop/Columbia/Courses/PML/Final Project/Data Collection/Data_for_Analysis/")
sentiments <- readRDS("tweet_sentiments.rds")
tweets <- readRDS("tweets_for_modeling.rds")

tweets <- tweets %>% left_join(sentiments, c("status_id"))

this_city <- "New York"
tweets_city <- tweets %>% filter(grepl(this_city, city)) %>%
                select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
m = (tweets_city %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s = tweets_city$score

data <- list(N=length(m), T=length(s), m=m, s=s, 
             eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))

vb_fit_ny <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                    tol_rel_obj=1e-3, importance_resampling=TRUE)

this_city <- "Boston"
tweets_city <- tweets %>% filter(grepl(this_city, city)) %>%
  select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
m = (tweets_city %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s = tweets_city$score

data <- list(N=length(m), T=length(s), m=m, s=s, 
             eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))

vb_fit_boston <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                    tol_rel_obj=1e-3,importance_resampling=TRUE, iter=500, eval_elbo=1)

elbo <- read.csv("elbo_multiple.csv", stringsAsFactors = FALSE)
ggplot(elbo[3:nrow(elbo),], aes(x=iter, y=ELBO_1)) + geom_line() + 
  ylab("Evidence Lower Bound (ELBO)") + xlab("Iteration") + 
  theme_classic()

elbo <- elbo %>% gather(key='run', value='ELBO',-iter)
ggplot(elbo, aes(x=iter, y=ELBO, col=run)) + geom_line() + 
  ylab("Evidence Lower Bound (ELBO)") + xlab("Iteration") + 
      theme_classic() + theme(legend.position = "none") 
ggsave("../../Writeup/figures/elbo.png", width = 12, height = 8,units="cm")

ny_samples <- extract(vb_fit_ny)
ny_alpha <- ny_samples$mu * ny_samples$nu
ny_beta  <- (1- ny_samples$mu) * ny_samples$nu
ny_distr <- rbeta(length(ny_alpha), ny_alpha, ny_beta)
ny_distr <- as.data.frame(ny_distr)
ggplot(ny_distr, aes(ny_distr)) + geom_density() + 
  xlab("User Attitudes") + ylab("Density") + 
    xlim(0,1) + theme_classic()

bos_samples <- extract(vb_fit_boston)
bos_alpha <- bos_samples$mu * bos_samples$nu
bos_beta  <- (1- bos_samples$mu) * bos_samples$nu
bos_distr <- rbeta(length(bos_alpha), bos_alpha, bos_beta)
bos_distr <- as.data.frame(bos_distr)
ggplot(bos_distr, aes(bos_distr)) + geom_density() + 
  xlab("User Attitudes") + ylab("Density") + 
  xlim(0,1) + theme_classic()

all_distr <- ny_distr
all_distr$city <- "New York, New York"
names(all_distr) <- c("distr", "city")
cities <- unique(tweets$city)[-1]

for(this_city in cities){
  tweets_city <- tweets %>% filter(this_city==city) %>%
    select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
  m = (tweets_city %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
  s = tweets_city$score
  
  data <- list(N=length(m), T=length(s), m=m, s=s, 
               eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))
  
  vb_fit <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                             tol_rel_obj=1e-3,importance_resampling=TRUE)
  
  samples <- extract(vb_fit)
  s_alpha <- samples$mu * samples$nu
  s_beta  <- (1- samples$mu) * samples$nu
  distr <- rbeta(length(s_alpha), s_alpha, s_beta)
  distr <- as.data.frame(distr)
  distr$city <- this_city
  all_distr <- rbind(all_distr, distr)
}

ggplot(all_distr %>% filter(grepl("Boston", city) | grepl("Chicago", city) ), aes(distr, col=city)) + 
  geom_density()

ggplot(all_distr %>% filter(grepl("California", city)), aes(distr, col=city)) + 
  geom_density()

ggplot(all_distr %>% filter(grepl("San", city) | grepl("Char", city) ), aes(distr, col=city)) + 
  geom_density()

ggplot(all_distr %>% filter(grepl("San Fran", city) | grepl("Los", city) ), aes(distr, col=city)) + 
  geom_density()

tweets %>% group_by(city) %>% summarise(users = n_distinct(screen_name),
                                        tweets = n()) %>% 
    arrange(-users) %>% filter(grepl("Cali", city))

plot_ca <- all_distr %>% filter(grepl("San Fran", city) | grepl("Los", city)) %>% 
              dplyr::rename(City=city) %>% 
              mutate(City = substr(City, 1, regexpr(",", City)-1))
ggplot(plot_ca, aes(distr, col=City)) + 
  geom_density() + xlab("Community Sentiment") + 
    ylab("Probability Density") + theme_classic() + 
      labs(col = "") + 
        theme(text= element_text(size=14), 
              axis.text.x = element_text(size=12), 
              axis.text.y = element_text(size=10), 
              legend.position=c(0.3,0.9))
ggsave(filename = "../../Report/Plots/cali.png", device=png(), 
       width = 6, height = 4, dpi="retina")

this_city <- "San Francisco"
tweets_city <- tweets %>% filter(grepl(this_city, city)) %>%
  select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
m = (tweets_city %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s = tweets_city$score
data <- list(N=length(m), T=length(s), m=m, s=s, 
             eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))
vb_fit_sf <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                           tol_rel_obj=1e-3,importance_resampling=TRUE)

this_city <- "Los Angeles"
tweets_city <- tweets %>% filter(grepl(this_city, city)) %>%
  select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
m = (tweets_city %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s = tweets_city$score
data <- list(N=length(m), T=length(s), m=m, s=s, 
             eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))
vb_fit_la <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                       tol_rel_obj=1e-3,importance_resampling=TRUE)

samples_sf <- extract(vb_fit_sf)
samples_la <- extract(vb_fit_la)

print(c(mean(samples_sf$mu), sd(samples_sf$mu)))
print(c(mean(1/samples_sf$nu), sd(1/samples_sf$nu)))

print(c(mean(samples_la$mu), sd(samples_la$mu)))
print(c(mean(1/samples_la$nu), sd(1/samples_la$nu)))

load(".Rdata")


ridge_cities <- (tweets %>% group_by(city) %>% 
                   summarise(users = n_distinct(screen_name)) %>% 
                   filter(users > 500))$city

ridge_subset <- all_distr %>% filter(city %in% ridge_cities)

ggplot(ridge_subset, aes(x=distr, y= reorder(factor(city), -distr, mean))) +
  geom_density_ridges(fill = "blue", alpha = .5, scale = 1, bandwidth=0.1) + 
  theme_classic() + xlim(0,1) + 
    theme(legend.title=element_blank()) + 
      xlab("Community Sentiment") + ylab("") +
          theme(text= element_text(size=14), 
                axis.text.x = element_text(size=12), 
                axis.text.y = element_text(size=10))

ggsave(filename = "../../Writeup/figures/cities_ridge.png", device=png(), 
       width = 6, height = 4, dpi="retina")

this_city <- "Los Angeles"
tweets_city <- tweets %>% filter(grepl(this_city, city)) %>%
  select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
m = (tweets_city %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s = tweets_city$score
data <- list(N=length(m), T=length(s), m=m, s=s, 
             eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))
vb_fit_la <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                       tol_rel_obj=1e-3,importance_resampling=TRUE)

this_city <- "San Diego"
tweets_ny <- tweets %>% filter(grepl(this_city, city)) %>%
                  select(screen_name, status_id, score) %>% arrange(screen_name) %>% na.omit()
train_idx <- sample(nrow(tweets_ny), nrow(tweets_ny)/2)
tweets_ny_train <- tweets_ny[train_idx,]
tweets_ny_test  <- tweets_ny[-train_idx,]

m = (tweets_ny_train %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s = tweets_ny_train$score

data <- list(N=length(m), T=length(s), m=m, s=s, 
             eta_mu=1, eta_nu=c(1, 1), eta_nu_s=c(1, 1))

vb_fit_ny <- rstan::vb(stan_model(file = '../../Analysis/tweet_model.stan'), data = data, 
                       tol_rel_obj=1e-3, importance_resampling=TRUE)

ny_samples <- extract(vb_fit_ny)
ny_alpha <- ny_samples$mu * ny_samples$nu
ny_beta  <- (1- ny_samples$mu) * ny_samples$nu
ny_distr <- rbeta(length(ny_alpha), ny_alpha, ny_beta)
ny_distr <- as.data.frame(ny_distr)

ny_samples$nu_s

m_test = (tweets_ny_test %>% group_by(screen_name) %>% summarise(n=n()) %>% select(n))$n
s_test = tweets_ny_test$score

ggplot(ny_distr, aes(ny_distr)) + geom_density() + 
  xlab("User Attitudes") + ylab("Density") + 
  xlim(0,1) + theme_classic()

s_pred <- c()
for(i in 1:length(m_test)){
  z_i <- sample(ny_distr$ny_distr,1)
  kappa <- sample(ny_samples$nu_s,1)
  
  z_alpha <- z_i * kappa
  z_beta  <- (1-z_i) * kappa
  
  s_pred <- c(s_pred, rbeta(m_test[i], z_alpha, z_beta))
}
posterior_pred <- as.data.frame(cbind(s_pred, s_test))
posterior_pred <- posterior_pred %>% gather(key='measure', value='tweet_sentiment')
ggplot(posterior_pred, aes(col=measure, x=tweet_sentiment)) + 
  geom_density() + theme_classic() + 
    xlab("Tweet Sentiment") + ylab("Density") +
      theme(text= element_text(size=14), 
            axis.text.x = element_text(size=12), 
            axis.text.y = element_text(size=10), 
            legend.position=c(0.7,0.8)) + 
          labs(col = "") + 
        scale_color_discrete(labels=c("Predicted", "Observed"))
        
ggsave(filename = "../../Writeup/figures/post_predictive.png", device=png(), 
       width = 9, height = 6, dpi="retina")
