function [gp_regr, nlml_regr, gp_class, nlml_class] = ...
    GPmodel(x_regr, y_regr, x_class, y_class, H_r, H_c, gp_no, corrErr)
% Fits and return GP regression and GP classification model to x and y
% GP1 regr: neural network cov fct (non-stationary) with optimization of hyperparam
% GP2 regr: sexp cov fct (stationary) with optimization of hyperparam
% GP3 regr: sexp cov fct (stationary) with optimization of hyperparam
% when HMC or variants of HMC used, where we require 1st, 2nd or 3rd gradient,
% we always use gp_no = 2 (squared exponential), as we calculate numerical
% gradients for the sq exp kernel

%% GP regression

% Hyperpriors
if corrErr == 0
    plg = prior_unif(); % prior for lengthscale
    pms = prior_sqrtunif();
    ps = prior_logunif(); % prior for sigma2 in the likelihood
    %lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', ps);
else
    %lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', prior_fixed);
%     plg = prior_sqrtt('s2', 100^2, 'nu', 3); % prior for lengthscale
%     pms = prior_sqrtt('s2', 100^2, 'nu', 3);
%     ps = prior_sqrtt('s2', 100^2, 'nu', 3);
    
    plg = prior_logunif();
    pms = prior_logunif();
    ps = prior_logunif();
end

% %Sample
% lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', ps);
% gpcf = gpcf_sexp('lengthScale', H_r(2:end-1), 'magnSigma2', H_r(1), ...
% 'lengthScale_prior', plg, 'magnSigma2_prior', pms);
% gp_regr = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4);
% [rgp_mcmc,g,opt]=gp_mc(gp_regr, x_regr, y_regr, 'nsamples', 1000, 'display', 20);
% rr_regr = thin(rgp_mcmc,500,2);
% gp_regr = rr_regr; nlml_regr=100;

for i = 1:size(H_r,1)
    lik = lik_gaussian('sigma2', H_r(end), 'sigma2_prior', ps);
    %lik = lik_t('sigma2', H_r(i,6), 'sigma2_prior', ps);
    
    % We allow for different lengthscale in every dimension (ARD)
    % One magnitude as it is a 1 single output (rss 1x1)
    
    if gp_no == 1
        % use non-stationary (neural network) cov fct for 1st GP
        gpcf = gpcf_neuralnetwork('biasSigma2', H_r(i,1), ...
            'weightSigma2', H_r(i,2:end-1), ...
            'biasSigma2_prior',prior_logunif(), ...
            'weightSigma2_prior',prior_logunif());
    else
        % use stationary (squared exponenetial) cov fct for 2nd & 3rd GP
        gpcf = gpcf_sexp('lengthScale', H_r(i,2:end-1), ...
            'magnSigma2', H_r(i,1),...
            'lengthScale_prior', plg, 'magnSigma2_prior', pms);
        
    end

    % Set a small amount of jitter to be added to the diagonal elements of the
    % covariance matrix K to avoid singularities when this is inverted
    jitter=1e-9;
    
    % Create the GP structure
    gp_regr_all{i} = gp_set('lik', lik, 'cf', gpcf,...
        'jitterSigma2', jitter);%, 'latent_method', 'EP');
    
%     % Sample
%     [rgp_mcmc,g,opt]=gp_mc(gp_regr_all{i}, x_regr, y_regr, ...
%     'nsamples', 300, 'display', 20);
%     rr_regr = thin(rgp_mcmc,100,2);
    
    % Set the options for the optimization
    opt=optimset('TolFun',1e-11,'TolX',1e-11);
    
    % Optimize with the scaled conjugate gradient method
    [gp_regr_all{i}, nlml_regr(i)] = ...
        gp_optim(gp_regr_all{i},x_regr,y_regr,'opt',opt);
    
end

I = find(nlml_regr == min(nlml_regr));
gp_regr = gp_regr_all{I};

disp('done')
%% GP classifier
lik_class = lik_logit();

for i = 1:size(H_c,1)
    % We allow for different lengthscale in every dimension (ARD)
    % One magnitude as it is a 1 single output (rss 1x1)
    gpcf = gpcf_sexp('lengthScale', H_c(i,2:end), ...
        'magnSigma2', H_c(i,1),...
        'lengthScale_prior', plg, 'magnSigma2_prior', pms); % sexp for wrong iid assumption on correlated errors
                                                            % MATERN32
                                                            % OTHERWISE
    % Create the GP structure (type is by default FULL)
    gp_class_all{i} = gp_set('lik', lik_class, 'cf', gpcf, ...
        'jitterSigma2', 1e-9);
    
    % ------- EP approximation --------
    % Set the approximate inference method - Laplace(default)
    gp_class_all{i} = gp_set(gp_class_all{i}, ...
        'latent_method', 'EP');
    
    % Set the options for the optimization
    opt=optimset('TolFun',1e-3,'TolX',1e-3);
    % Optimize with the scaled conjugate gradient method
    [gp_class_all{i}, nlml_class(i)] = ...
        gp_optim(gp_class_all{i}, x_class,y_class,'opt',opt);
    %, 'optimf', @fminlbfgs);
end

I = find(nlml_class == min(nlml_class));
gp_class = gp_class_all{I};

end

