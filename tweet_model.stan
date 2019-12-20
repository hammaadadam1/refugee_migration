// saved as 8schools.stan
data {
  int<lower=0> N;         // number of people
  int<lower=0> T;         // total number of tweets
  int m[N];               // Number of tweets per person
  real<lower=0> s[T];     // Sentiment of individual tweets
  
  real<lower=0> eta_mu;
  vector<lower=0>[2] eta_nu;
  vector<lower=0>[2] eta_nu_s;
}
parameters {
  real<lower=0, upper=1> mu;      
  real<lower=0> nu;
  real<lower=0> nu_s;
  vector<lower=0, upper=1>[N] z;   
}
transformed parameters {
  vector<lower=0>[N] alpha_z;
  vector<lower=0>[N] beta_z;
  
  alpha_z = rep_vector(mu*nu, N);
  beta_z = rep_vector(nu*(1- mu), N);
  
}
model {
  int pos;
  
  mu ~ beta(eta_mu,eta_mu);  // prior
  nu ~ gamma(eta_nu[1], eta_nu[2]);
  
  pos=1;
  for(i in 1:N){
    z[i] ~ beta(alpha_z[i], beta_z[i]);
    segment(s, pos, m[i]) ~ beta(nu_s*z[i], nu_s*(1-z[i]));
    pos = pos + m[i];
  }
}

