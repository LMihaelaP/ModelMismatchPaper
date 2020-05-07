function gp_regr = ReFitGP(x_regr, y_regr, corrErr)
% Fits and return GP regression  model to x and y

%% GP regression

% Hyperpriors
if corrErr == 0
    plg = prior_unif(); % prior for lengthscale
    pms = prior_sqrtunif();
    ps = prior_logunif(); % prior for sigma2 in the likelihood
    %lik = lik_gaussian('sigma2', exp(-9.950929526405522), 'sigma2_prior', ps);
else
    %lik = lik_gaussian('sigma2', 4.7089e-08, 'sigma2_prior', prior_fixed);
%     plg = prior_sqrtt('s2', 100^2, 'nu', 3); % prior for lengthscale
%     pms = prior_sqrtt('s2', 100^2, 'nu', 3);
%     ps = prior_sqrtt('s2', 100^2, 'nu', 3);
    
    plg = prior_logunif();
    pms = prior_logunif();
    ps = prior_logunif();
end

% %Sample
% lik = lik_gaussian('sigma2', 1.1, 'sigma2_prior', ps);
% gpcf = gpcf_sexp('lengthScale', [1 1 1 1], 'magnSigma2', 10^6, ...
% 'lengthScale_prior', plg, 'magnSigma2_prior', pms);
% gp_regr = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4);
% [rgp_mcmc,g,opt]=gp_mc(gp_regr, x_regr, y_regr, 'nsamples', 400, 'display', 20);
% gp_regr = thin(rgp_mcmc,100,2);


% We allow for different lengthscale in every dimension (ARD)
% One magnitude as it is a 1 single output (rss 1x1)
lik = lik_gaussian('sigma2', exp(-9.950929526405522), 'sigma2_prior', ps);

if corrErr == 0
    gpcf = gpcf_sexp('lengthScale', exp([-2.865405448983861 -1 -1.465713776071921 ...
        -2.523119703681556 -0.421869887754162]), 'magnSigma2', 1,...
        'lengthScale_prior', plg, 'magnSigma2_prior', pms);
else
    
    gpcf = gpcf_sexp('lengthScale', 0.01*[1 1 1 1 1 1 1], ...
        'magnSigma2', 1,...
        'lengthScale_prior', plg, 'magnSigma2_prior', pms);
end
% Set a small amount of jitter to be added to the diagonal elements of the
% covariance matrix K to avoid singularities when this is inverted
jitter=1e-9;

% Create the GP structure
gp_regr = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', jitter);

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);

% Optimize with the scaled conjugate gradient method
gp_regr = gp_optim(gp_regr,x_regr,y_regr,'opt',opt);

end
