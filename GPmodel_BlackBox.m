function gp = GPmodel_BlackBox(x, y, l, u)
% function [gp, gp_classif] = GPmodel_BlackBox(x, y, l, u, x_classif, y_classif)
% Fits GP model for the objective function:
% normalised expected squared distance  used in Bayesian Optimization

lik = lik_gaussian('sigma2', 0.1, 'sigma2_prior', prior_fixed);
%lik = lik_t('nu', 4, 'nu_prior', ps, 'sigma2', 0.1, 'sigma2_prior', ps);

alpha = 0.2;

lgt1 = (alpha * (u(1)-l(1))); % lengthscale for epsilon
lgt2 = (alpha * (u(2)-l(2))); % lengthscale for L

gpcf = gpcf_sexp('lengthScale', [lgt1, lgt2], 'magnSigma2', 1,...
    'lengthScale_prior', prior_fixed, 'magnSigma2_prior', prior_fixed);

% gpcf = gpcf_neuralnetwork('biasSigma2', 1, ...
%             'weightSigma2', [lgt1, lgt2], ...
%             'biasSigma2_prior',prior_logunif(), ...
%             'weightSigma2_prior',prior_logunif());
        
% %Sample gp hyperparameters using mcmc
% gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4);
% rng('shuffle')
% gp_mcmc = gp_mc(gp, x, y, 'nsamples', 150, 'display', 50);
% gp = thin(gp_mcmc,149,2);

% % Optimise hyperparameters
jitter=1e-9;

gp = gp_set('lik', lik, 'cf', gpcf,...
    'jitterSigma2', jitter);
% 
% opt=optimset('TolFun',1e-3,'TolX',1e-3);
% 
% gp = gp_optim(gp,x,y,'opt',opt);




% gpcf = gpcf_matern32('lengthScale', [lgt1, lgt2], 'magnSigma2', 10,...
%     'lengthScale_prior', plg, 'magnSigma2_prior', pms);
%
% %Sample gp hyperparameters using mcmc
% gp_classif = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4);
% rng default
% gp_mcmc_classif = gp_mc(gp_classif, x_classif, y_classif, 'nsamples', 100, 'display', 20);
% gp_classif = thin(gp_mcmc_classif,25,2);

end

