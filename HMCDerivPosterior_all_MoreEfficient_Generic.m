function [LogPosterior, GradLogPosterior, GradGradLogPosterior, ...
    GradGradGradLogPosterior, Var, ObjFct] = ...
    HMCDerivPosterior_all_MoreEfficient_Generic(param_sc, ...
    sigma2, truePressure, alp, bet, GP_hyperHyper, extra_p, l, u, sc, ...
    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr, x_regr, y_regr, gp_class, x_class, y_class, mean_y, std_y, ...
    do_nuts, corrErr, gp_ind)
% Function that calculates the LogPosterior distr and its gradient for both
% the simulator (needed at end of trajectory) - when em_ind = 0
% and the emulator (needed throughout the trajectory) - when em_ind = 1
% Also returns Var from emulator to check if sqrt(Var) > 3,
% i.e. stop the trajectory in HMC?

% Input
% param_sc: (nd,1) vector -- parameters scaled to make them unbounded
% alp, bet: (nd,1) vectors -- hyperparameters for the Beta distr that the bio prior follows
% l, u: lower and upper bound for the original bio & GPhyper parameters
% sc: the constant scaling factors used in emulation: [10^6,10,10,10,1,10,0.1]
% em_ind: emulation index that indicates if we simulate (0) or emulate (1)
% the objective function (RSS if no error correlation, loglik if error correlation)
% phase_ind: GPHMC algthm phase index - 1: exploratory, 2: sampling phase
%            as E_potential different for the 2 phases
% grad1_SimInd: indicates if we calculate 1st gradient of the log posterior
% coming from the simulator (can be 0 at the end of the trajectory
% when we accept/reject based on MH step, ie no GradLogPost needed; can be 1 if
% we use the simulator throughout the trajectory, ie GradLogPost needed).
% grad23_EmInd: 2x1 indicator vector: says if we calculate 2nd & 3rd gradient of the log posterior
% coming from the emulator (only done in the sampling phase),
% i.e. whether we do HMC/NUTS or RMHMC/LMC/BAHMC
% gp_regr, x_regr, y_regr, gp_class, x_class, y_class: GP regression and
% classification used throughout the trajectory
% mean_y, std_y: used to make y_regr zero mean, variance 1
% do_nuts: indicator if NUTS is done or not
% corrErr: indicator whether we use code for correlated errors
% gp_ind: indicator for which covariance function we will use if correlated
% errors code is used

% Output
% LogPosterior, 1st, 2nd, 3rd Gradient LogPosterior
% Var to check if sqrt(Var) > 3, i.e. stop the trajectory in HMC?
% ObjFct value: RSS if corrErr = 0
%           loglik (incorporates covariance matrix for errors) if corrErr=1

nd = extra_p(2); % parameter dimensionality
hd = length(GP_hyperHyper)/2; % 2 gp hyperparameters
nbio = nd-hd;

param = NaN(1,nd);

n = numel(truePressure);

if do_nuts == 1
    param_sc = param_sc';
end

expPar = exp(param_sc);

if corrErr == 1
    
    Sign = 1; % indicates whether E + or - sqrt(Var) for the objective function in phase_ind = 1 (-1 for rss and 1 for loglik)
    
    % bio parameters on original scale
    param(1:nbio) = (u(1:nbio).*expPar(1:nbio)+l(1:nbio))./(1+expPar(1:nbio));
    % GP hyperparameters on original scale
    if gp_ind == 5 % neural network
        param(end-1:end) = (u(end-1:end).*expPar(end-1:end)+l(end-1:end))./(1+expPar(end-1:end));
        %param(7) = expPar(end);
    else % all the other cov fcts
        param(end-1:end) = expPar(end-1:end);
    end
    
    if gp_ind ~= 5
        Jacob = [((expPar(1:nbio).*(u(1:nbio)-l(1:nbio)))./ (1+expPar(1:nbio)).^2),...
            param(end-1), param(end)];
    else
        Jacob = [((expPar(1:nbio).*(u(1:nbio)-l(1:nbio)))./ (1+expPar(1:nbio)).^2),...
            ((expPar(end-1:end).*(u(end-1:end)-l(end-1:end)))./ (1+expPar(end-1:end)).^2)];% is this param or param_sc?
    end
    
else
    
    Sign = -1;
    
    param = (u.*expPar+l)./(1+expPar); % parameters on original scale
    
    Jacob = ((expPar.*(u-l))./ (1+expPar).^2);
end


if corrErr == 0
    
    
    if em_ind == 0
        param_sc;
        (u.*expPar+l)./(1+expPar);
        
        if any(isnan(param)) || any(~isfinite(param))
            %disp('The LogLikGradient is too high, change values of L and epsilon')
            ObjFct = 10^10; pass = 0;
        else
            
            [ObjFct, pass, Z, ~, ~, pressure] = Run_simulator(param./sc, extra_p, truePressure, sc, ...
                gp_ind, corrErr);
            
        end
        
        % Compute the simulated log likelihood and its gradient
        LogLik = -n/2*log(sigma2)-n/2*log(2*pi) - ObjFct/(2*sigma2);
        
        %     if cx == 0
        %         sorFirstDeriv_phi = BlackBoxFirstDeriv_phi(pressure, param_sc, extra_p, l, u);
        %     else
        %         sorFirstDeriv_phi = zeros(nd,n);
        %     end
        %     GradLogLik = (1/sigma2) * sorFirstDeriv_phi * Z;
        
        if grad1_SimInd == 1
            if pass == 1
                
                sorFirstDeriv_theta = BlackBoxFirstDeriv_theta(pressure, param, extra_p);
                
            else
                
                sorFirstDeriv_theta = zeros(nd,n);
            end
            
            GradLogLik = (1/sigma2) * (sorFirstDeriv_theta * Z) .* Jacob';
            
        else % grad1_SimInd = 0
            
            GradLogLik = NaN(nd,1);
            
        end
        
        % Assign Var of GP prediction Inf since here simulator used
        Var = Inf;
        
    else % em_ind == 1
        %%%%%%%%%%%% here compute emulated loglik and derivative of emulated loglik
        %%%%%%
        
        param_em =  param./sc; % parameters used in emulation
        
        if any(isnan(param_em)) || any(~isfinite(param_em))
            %disp('The LogLikGradient is too high, change values of L and epsilon')
            ObjFct = 10^10; pass = 0; Var = Inf;
            
        else
            
            [E,Var] = gp_pred(gp_regr, x_regr, y_regr, param_em);
            %                     E = sek * a;
            %                     v = L\sek';
            %                     Var = sexp_k(param_em, param_em) - v'*v;
            %                     %or Var = sexp_k(param_em, param_em) -
            %                     %sek*(inv(L))'*inv(L)*sek';
            
            if phase_ind == 1 % exploratory phase
                ObjFct = (E + Sign * sqrt(Var)) * std_y + mean_y; % E-sqrt(Var)
            else % phase_ind == 2 => sampling phase
                ObjFct = E * std_y + mean_y;
            end
            
            
            [~, ~, lpyt_la, ~, ~] = gp_pred(gp_class, x_class, ...
                y_class, param_em, 'yt', 1);
            
            if exp(lpyt_la) < 0.8 % unsuccessful simulation
                ObjFct = 1e+10; % very large value for RSS for unsuccessful simulation
                pass = 0;
            else
                pass = 1;
            end
            
        end
        
        % Now compute the emulated log likelihood and its gradient
        LogLik = -n/2*log(sigma2)-n/2*log(2*pi) - ObjFct/(2*sigma2);
        
        if pass == 0
            
            GradLogLik = -(1/(2*sigma2)) * zeros(nd,1);
            
        else
            
            [~,Cov] = gp_trcov(gp_regr, x_regr);
            L = chol(Cov,'lower');
            a = L'\(L\y_regr); % (inv(L))'*(inv(L)*y_regr); slower
            %a = inv(C) * y_regr; slower
            
            magnS2 = gp_regr.cf{1}.magnSigma2; % (1,1)
            lgtScales = gp_regr.cf{1}.lengthScale; % (1,d)
            Lambda = diag(1./(lgtScales.^2)); % (d,d)
            
            % Construct: magnS2 * exp(-0.5*((x-y)*Lambda*(x-y)'))
            AT = ( repmat(param_em,size(x_regr,1),1) - x_regr ) * Lambda; % (n,d)
            BT = ( repmat(param_em,size(x_regr,1),1) - x_regr )'; % (d,n)
            
            CT = sum(AT.*BT',2)'; % (1,n)
            
            sek = magnS2 .* exp(-0.5.*CT); % (1,n)
            
            q1 = Lambda * BT; % (d,n)
            %q2 = sek' .* a;
            q4 = 1/(2*sigma2) * std_y; % (1,1)
            
            ScJacob = ((1./sc)' .* Jacob'); % scaled Jacob (d,1)
            
            FirstDerivKernel = - q1 .* sek; % (d,n) since (d,n) .* (1,n) = (d,n)
            
            FirstDerivGPpostMean = FirstDerivKernel * a; % first order derivative of the GP posterior mean (d,1)
            
            GradLogLik = (-q4 * FirstDerivGPpostMean) .* ScJacob; % (d,1)
            
            if phase_ind == 1
                
                v = L\sek';
                q5 = (v'/L);
                
                FirstDerivGPpostVar = -2 * q5 * FirstDerivKernel';
                FirstDerivGPpostVar = FirstDerivGPpostVar'; %(d,1)
                
                GradLogLik = GradLogLik + ( -q4 * Sign * 1/(2*sqrt(Var)) * ...
                    FirstDerivGPpostVar ) .* ScJacob; % (d,1)
                
            end % phase_ind
            
            if grad23_EmInd(1) == 1 % 2nd order derivative of the emulated Log Lik
                
                SecondDerivKernel = - Lambda * ( ones(nd,1) * sek + BT .* FirstDerivKernel ); % (d,n)
                
                SecondDerivGPpostMean = SecondDerivKernel * a; % second order derivative of the GP posterior mean (d,1)
                
                dJacob = (((expPar.*(1-expPar) .* (u-l)))./ (1+expPar).^3); % first order deriv of Jacob (1,d)
                
                ScdJacob = ((1./sc) .* dJacob)'; % scaled first order deriv of Jacob (d,1)
                
                Q = ScJacob .* SecondDerivGPpostMean; % (d,1)
                
                GradGradLogLik = -q4 *...
                    ( (Q .* ScJacob ) + FirstDerivGPpostMean .* ScdJacob ); % (d,1)
                
                
                if grad23_EmInd(2) == 1 % 3rd order derivative of the emulated Log Lik
                    % (RMHMC, LDMC)
                    
                    ThirdDerivKernel = - Lambda * ( 2 * ones(nd,size(x_regr,1)) .* FirstDerivKernel + BT .* SecondDerivKernel  ); %(d,n)
                    
                    ThirdDerivGPpostMean = ThirdDerivKernel * a; % third order derivative of the GP posterior mean (d,1)
                    
                    
                    d2Jacob = ( ( (u-l).*expPar.*((1+expPar).^2).* ...
                        ( (expPar-2).^2 - 3) ) ./ (1+expPar).^6 ); % (1,d)
                    
                    Scd2Jacob = ((1./sc) .* d2Jacob )'; % scaled second order deriv of Jacob (d,1)
                    
                    R = ThirdDerivGPpostMean .* ScJacob .* ScJacob + SecondDerivGPpostMean .* ScdJacob; % (d,1)
                    
                    
                    GradGradGradLogLik = -q4 *...
                        ( ( R .* ScJacob ) + 2 .* ( Q .* ScdJacob ) + ...
                        (FirstDerivGPpostMean .* Scd2Jacob) ); % (d,1)
                    
                end % grad23_EmInd(2)
                
            end % grad23_EmInd(1)
            
        end %ObjFCt
        
    end % em_ind
    
    
    % Compute log priors for the scaled parameters and their gradients
    LogPrior = sum( alp.*param_sc - (alp+bet).* log(1+expPar) - ...
        betaln(alp,bet) );
    
    if em_ind == 0 && grad1_SimInd == 0
        GradLogPrior = NaN(nd,1);
    else
        
        % first derivative
        GradLogPrior = (alp - bet .* expPar)./(1+expPar) ;
        GradLogPrior = GradLogPrior';
    end
    
    
    if em_ind == 1 && grad23_EmInd(1) == 1
        
        % second derivative of the prior
        GradGradLogPrior = -((alp+bet).* expPar)./(1+expPar).^2;
        GradGradLogPrior = GradGradLogPrior';
        
        if grad23_EmInd(2) == 1
            
            % third derivative of the prior
            GradGradGradLogPrior = -((alp+bet).*(expPar.*(1-expPar))./(1+expPar).^3);
            GradGradGradLogPrior = GradGradGradLogPrior';
            
        end % grad23_EmInd(2)
        
    end % em_ind && grad23_EmInd(1)
    
else % corrErr = 1
    
    if em_ind == 0
        param_sc;
        (u(1:nbio).*expPar(1:nbio)+l(1:nbio))./(1+expPar(1:nbio));
        
        if any(isnan(param)) || any(~isfinite(param))
            %disp('The LogLikGradient is too high, change values of L and epsilon')
            ObjFct = -10^10; pass = 0;
            
        else
            
            [ObjFct, pass, Z, C, L, pressure] = Run_simulator(param./sc, ...
                extra_p, truePressure, sc, gp_ind, corrErr);
            
        end
        
        % Compute the simulated log likelihood and its gradient
        LogLik = ObjFct;
        
        if grad1_SimInd== 1
            if pass == 1
                
                sorFirstDeriv_theta = BlackBoxFirstDeriv_theta(pressure, param(1:nbio), [extra_p(1) nbio]);
                
            else
                
                sorFirstDeriv_theta = zeros(nbio,n);
            end
            
            LZtemp = L\Z;
            
            GradLogLik_bioPar = (1/2) .* ( ((sorFirstDeriv_theta/L') * LZtemp) + ...
                (LZtemp' * (L\sorFirstDeriv_theta'))' ) .* Jacob(1:nbio)';
            
            CovMatrix1stDeriv = HMCDerivLogLik_GPhyper(param, gp_ind, ...
                pressure, truePressure, l, u);
            
            GradLogLik_GPhyperPar = NaN(hd,1);
            
            for d = 1:hd
                GradLogLik_GPhyperPar(d) = -0.5*trace(C\CovMatrix1stDeriv(:,:,d)) + ...
                    0.5*(Z'/C)*CovMatrix1stDeriv(:,:,d)*(C\Z);
            end
            
            
            GradLogLik = [GradLogLik_bioPar; GradLogLik_GPhyperPar'];
            
            %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %     derivLogPost = @(theta)Pulm_GradientChecking(theta, extra_p, ...
            %         truePressure, l,u,sc, gp_ind, corrErr);
            %
            %     mycheckderivative(param_sc, derivLogPost, 1e-3);
            %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        else
            
            GradLogLik = NaN(nd,1);
            
        end %grad1_SimInd
        
        % Assign Var of GP prediction Inf since here simulator used
        Var = Inf;
        
    else % em_ind == 1
        %%%%%%%%%%%% here compute emulated loglik and derivative of emulated loglik
        %%%%%%
        
        param_em =  param./sc; % parameters used in emulation
        
        if any(isnan(param_em)) || any(~isfinite(param_em))
            %disp('The LogLikGradient is too high, change values of L and epsilon')
            ObjFct = -10^10; pass = 0; Var = Inf;
            
        else
            
            [E,Var] = gp_pred(gp_regr, x_regr, y_regr, param_em);
            %                     E = sek * a;
            %                     v = L\sek';
            %                     Var = sexp_k(param_em, param_em) - v'*v;
            %                     %or Var = sexp_k(param_em, param_em) -
            %                     %sek*(inv(L))'*inv(L)*sek';
            
            if phase_ind == 1 % exploratory phase
                ObjFct = (E + Sign * sqrt(Var)) * std_y + mean_y; % E+sqrt(Var)
            else % phase_ind == 2 => sampling phase
                ObjFct = E * std_y + mean_y;
            end
            
            
            [~, ~, lpyt_la, ~, ~] = gp_pred(gp_class, x_class, ...
                y_class, param_em, 'yt', 1);
            
            if exp(lpyt_la) < 0.8 % unsuccessful simulation
                ObjFct = -1e+10; % very small value for loglik for unsuccessful simulation
                pass = 0;
            else
                pass = 1;
            end
            
        end
        
        % Now compute the emulated log likelihood and its gradient
        LogLik = ObjFct;
        
        if pass == 0
            
            GradLogLik = zeros(nd,1);
            
        else
            
            [~,Cov] = gp_trcov(gp_regr, x_regr);
            L = chol(Cov,'lower');
            a = L'\(L\y_regr); % (inv(L))'*(inv(L)*y_regr);
            %a = inv(C) * y_regr;
            
            magnS2 = gp_regr.cf{1}.magnSigma2; % (1,1)
            lgtScales = gp_regr.cf{1}.lengthScale; % (1,d)
            Lambda = diag(1./(lgtScales.^2)); % (d,d)
            
            % Construct: magnS2 * exp(-0.5*((x-y)*Lambda*(x-y)'))
            AT = ( repmat(param_em,size(x_regr,1),1) - x_regr ) * Lambda; % (n,d)
            BT = ( repmat(param_em,size(x_regr,1),1) - x_regr )'; % (d,n)
            
            CT = sum(AT.*BT',2)'; % (1,n)
            
            sek = magnS2 .* exp(-0.5.*CT); % (1,n)
            
            q1 = Lambda * BT; % (d,n)
            
            ScJacob = ((1./sc)' .* Jacob'); % scaled Jacob (d,1)
            
            FirstDerivKernel = - q1 .* sek; % (d,n) since (d,n) .* (1,n) = (d,n)
            
            FirstDerivGPpostMean = FirstDerivKernel * a; % first order derivative of the GP posterior mean (d,1)
            
            GradLogLik = std_y * FirstDerivGPpostMean .* ScJacob; % (d,1)
            
            
            if phase_ind == 1 % exploratory phase
                %disp('exploratory phase')
                v = L\sek';
                q5 = (v'/L);
                
                FirstDerivGPpostVar = -2 * q5 * FirstDerivKernel';
                FirstDerivGPpostVar = FirstDerivGPpostVar'; %(d,1)
                
                GradLogLik = GradLogLik + ( std_y * Sign * ...
                    1/(2*sqrt(Var)) * (FirstDerivGPpostVar ) ).* ScJacob;  % (d,1)
                
            end % phase_ind
            
            
            if grad23_EmInd(1) == 1
                
                SecondDerivKernel = - Lambda * ( ones(nd,1) * sek + BT .* FirstDerivKernel ); % (d,n)
                
                SecondDerivGPpostMean = SecondDerivKernel * a; % second order derivative of the GP posterior mean (d,1)
                
                Q = ScJacob .* SecondDerivGPpostMean; % (d,1)
                
                % Calculate 1st gradient of Jacobian
                if gp_ind ~= 5
                    dJacob = [(((expPar(1:nbio).*(1-(expPar(1:nbio))).*...
                        (u(1:nbio)-l(1:nbio))))./ (1+expPar(1:nbio)).^3), ...
                        param(end-1), param(end)]; % (1,d)
                    
                else % neural network
                    dJacob = (((expPar(1:end).*(1-(expPar(1:end))).*...
                        (u(1:end)-l(1:end))))./ (1+expPar(1:end)).^3); % (1,d)
                    
                end
                
                ScdJacob = ((1./sc) .* dJacob)'; % scaled first order deriv of Jacob (d,1)
                
                GradGradLogLik = std_y * ...
                    ( (Q .* ScJacob ) + FirstDerivGPpostMean .* ScdJacob ); % (d,1)
                
                
                if grad23_EmInd(2) == 1
                    
                    ThirdDerivKernel = - Lambda * ( 2 * ones(nd,size(x_regr,1)) .* FirstDerivKernel + BT .* SecondDerivKernel  ); %(d,n)
                    
                    ThirdDerivGPpostMean = ThirdDerivKernel * a; % third order derivative of the GP posterior mean (d,1)
                    
                    if gp_ind ~= 5
                        
                        d2Jacob = [( ( (u(1:nbio)-l(1:nbio)).*expPar(1:nbio).*((1+expPar(1:nbio)).^2).* ...
                            ( (expPar(1:nbio)-2).^2 - 3) ) ./ (1+expPar(1:nbio)).^6 ), ...
                            param(end-1), param(end)]; % (1,d)
                        
                    else % neural network
                        
                        d2Jacob = ( ( (u(1:end)-l(1:end)).*expPar(1:end).*((1+expPar(1:end)).^2).* ...
                            ( (expPar(1:end)-2).^2 - 3) ) ./ (1+expPar(1:end)).^6 ); % (1,d)
                    end
                    
                    Scd2Jacob = ((1./sc) .* d2Jacob )'; % scaled second order deriv of Jacob (d,1)
                    
                    
                    R = ThirdDerivGPpostMean .* ScJacob .* ScJacob + SecondDerivGPpostMean .* ScdJacob; % (d,1)
                    
                    
                    GradGradGradLogLik = std_y * ( ( R .* ScJacob ) + ...
                        2 .* ( Q .* ScdJacob ) + (FirstDerivGPpostMean .* Scd2Jacob) ); % (d,1)
                    
                end % grad23_EmInd(2)
                
            end %grad23_EmInd(1)
            %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %             derivLogPost = @(theta)Pulm_GradientChecking(theta, extra_p, truePressure, l,u,sc, nd,...
            %             gp_ind, corrErr,phase_ind,gp_regr,x_regr,y_regr,gp_class,x_class,...
            %             y_class,mean_y,std_y);
            %
            %             mycheckderivative(param_sc, derivLogPost, 1e-1);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end %ObjFCt
        
    end % em_ind
    
    
    %Compute log priors for the scaled parameters and their gradients
    % For biological parameters
    LogPrior_bioPar = sum( alp.*param_sc(1:nbio) - (alp+bet).* log(1+expPar(1:nbio)) - ...
        betaln(alp,bet) );
    
    if em_ind == 1 || grad1_SimInd == 1
        GradLogPrior_bioPar = (alp - bet .* expPar(1:nbio))./(1+expPar(1:nbio)) ;
    end
    
    if em_ind == 1 && grad23_EmInd(1) == 1 % 2nd order deriv of the prior
        
        GradGradLogPrior_bioPar = -((alp+bet).* expPar(1:nbio))./(1+expPar(1:nbio)).^2;
        
        if grad23_EmInd(2) == 1 % 3rd order deriv of the prior
            
            GradGradGradLogPrior_bioPar = -((alp+bet).*((expPar(1:nbio)).*...
                (1-expPar(1:nbio)))./(1+expPar(1:nbio)).^3);
            
        end % grad23_EmInd(2)
        
    end %em_ind && grad23_EmInd(1)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     derivLogPost = @(theta)Pulm_GradientChecking(theta, alp,bet);
    %
    %     mycheckderivative(param_sc, derivLogPost, 1e-5);
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % For GP covariance hyperparameters
    if gp_ind ~= 5
        LogPrior_GPhyperPar = - (log(sqrt(GP_hyperHyper(2))) + 0.5*log(2*pi)) - ...
            ((param_sc(end-1) - GP_hyperHyper(1))^2)/(2*GP_hyperHyper(2)) + ... %
            - (log(sqrt(GP_hyperHyper(4))) + 0.5*log(2*pi)) - ...
            ((param_sc(end) - GP_hyperHyper(3))^2)/(2*GP_hyperHyper(4));% + ... %
        %- (log(sqrt(GP_hyperHyper(6))) + 0.5*log(2*pi)) - ...
        %((param_sc(7) - GP_hyperHyper(5))^2)/(2*GP_hyperHyper(6));
        
        %log(GP_hyperHyper(6)^GP_hyperHyper(5)/gamma(GP_hyperHyper(5))) - ...
        %(GP_hyperHyper(5)+1)*param_sc(7) - GP_hyperHyper(6)/exp(param_sc(7)) + param_sc(7);
        
        
        %
        % 1st derivative
        if em_ind == 1 || grad1_SimInd == 1
            
            GradLogPrior_GPhyperPar = NaN(1,hd);
            
            GradLogPrior_GPhyperPar(1) =  - (param_sc(end-1) - GP_hyperHyper(1))/GP_hyperHyper(2);
            GradLogPrior_GPhyperPar(2) =  - (param_sc(end) - GP_hyperHyper(3))/GP_hyperHyper(4);
            %GradLogPrior_GPhyperPar(3) =  - (param_sc(7) - GP_hyperHyper(5))/GP_hyperHyper(6);
            
            %GradLogPrior_GPhyperPar(3) = GP_hyperHyper(6)/exp(param_sc(7)) - GP_hyperHyper(5);
        end
        
        %
        if em_ind == 1 && grad23_EmInd(1) == 1
            % 2nd derivative
            GradGradLogPrior_GPhyperPar = NaN(1,hd);
            
            GradGradLogPrior_GPhyperPar(1) =  - 1/GP_hyperHyper(2);
            GradGradLogPrior_GPhyperPar(2) =  - 1/GP_hyperHyper(4);
            %GradGradLogPrior_GPhyperPar(3) =  - 1/GP_hyperHyper(6);
            
            %GradGradLogPrior_GPhyperPar(3) = - GP_hyperHyper(6)/exp(param_sc(7));
            %
            
            if grad23_EmInd(2) == 1
                % 3rd derivative
                GradGradGradLogPrior_GPhyperPar = NaN(1,hd);
                
                GradGradGradLogPrior_GPhyperPar(1) = 0;
                GradGradGradLogPrior_GPhyperPar(2) = 0;
                %GradGradGradLogPrior_GPhyperPar(3) = 0;
                
                %GradGradGradLogPrior_GPhyperPar(3) = GP_hyperHyper(6)/exp(param_sc(7));
                
            end % grad23_EmInd(2)
            
        end % em_ind && grad23_EmInd(1)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         derivLogPost = @(theta)Pulm_GradientChecking(theta, GP_hyperHyper);
        %
        %         mycheckderivative(param_sc, derivLogPost, 1e-5);
        %         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    else % gp_ind == 5
        LogPrior_GPhyperPar = log((1+expPar(end-1))/...
            ((GP_hyperHyper(2)*expPar(end-1)+GP_hyperHyper(1))*...
            (log(GP_hyperHyper(2))-log(GP_hyperHyper(1)))))+ ...
            log((expPar(end-1)*(GP_hyperHyper(2)-GP_hyperHyper(1)))/...
            (1+expPar(end-1))^2) + ... %
            log((1+expPar(end))/...
            ((GP_hyperHyper(4)*expPar(end)+GP_hyperHyper(3))*...
            (log(GP_hyperHyper(4))-log(GP_hyperHyper(3)))))+ ...
            log((expPar(end)*(GP_hyperHyper(4)-GP_hyperHyper(3)))/...
            (1+expPar(end))^2);% + ... %
        %- (log(sqrt(GP_hyperHyper(6))) + 0.5*log(2*pi)) - ...
        %((param_sc(7) - GP_hyperHyper(5))^2)/(2*GP_hyperHyper(6));
        
        %log(GP_hyperHyper(6)^GP_hyperHyper(5)/gamma(GP_hyperHyper(5))) - ...
        %(GP_hyperHyper(5)+1)*param_sc(7) - GP_hyperHyper(6)/exp(param_sc(7)) + param_sc(7);
        %
        % 1st derivative
        if em_ind == 1 || grad1_SimInd == 1
            
            GradLogPrior_GPhyperPar = NaN(1,hd);
            
            GradLogPrior_GPhyperPar(1) = -expPar(end-1)/(1+expPar(end-1)) - ...
                (GP_hyperHyper(2)*expPar(end-1))/...
                (GP_hyperHyper(2)*expPar(end-1)+GP_hyperHyper(1)) + 1;
            
            GradLogPrior_GPhyperPar(2) = -expPar(end)/(1+expPar(end)) - ...
                (GP_hyperHyper(4)*expPar(end))/...
                (GP_hyperHyper(4)*expPar(end)+GP_hyperHyper(3)) + 1;
            
            %GradLogPrior_GPhyperPar(3) =  - (param_sc(7) - GP_hyperHyper(5))/GP_hyperHyper(6);
            
            %GradLogPrior_GPhyperPar(3) = GP_hyperHyper(6)/exp(param_sc(7)) - GP_hyperHyper(5);
        end
        %
        if em_ind == 1 && grad23_EmInd(1) == 1
            % 2nd derivative
            GradGradLogPrior_GPhyperPar = NaN(1,hd);
            
            GradGradLogPrior_GPhyperPar(1) = -expPar(end-1)/((1+expPar(end-1))^2) - ...
                (GP_hyperHyper(1)*GP_hyperHyper(2)*expPar(end-1))/...
                ((GP_hyperHyper(2)*expPar(end-1)+GP_hyperHyper(1))^2);
            
            GradGradLogPrior_GPhyperPar(2) = -expPar(end)/((1+expPar(end))^2) - ...
                (GP_hyperHyper(3)*GP_hyperHyper(4)*expPar(end))/...
                ((GP_hyperHyper(4)*expPar(end)+GP_hyperHyper(3))^2);
            
            %GradGradLogPrior_GPhyperPar(3) =  - 1/GP_hyperHyper(6);
            
            %GradGradLogPrior_GPhyperPar(3) = - GP_hyperHyper(6)/exp(param_sc(7));
            %
            
            if grad23_EmInd(2) == 1
                % 3rd derivative
                GradGradGradLogPrior_GPhyperPar = NaN(1,hd);
                
                GradGradGradLogPrior_GPhyperPar(1) = -(expPar(end-1)*(1-expPar(end-1)))/...
                    ((1+expPar(end-1))^3) - ...
                    ( GP_hyperHyper(1)*GP_hyperHyper(2)*expPar(end-1)*...
                    (GP_hyperHyper(1)-GP_hyperHyper(2)*expPar(end-1)) )/...
                    ((GP_hyperHyper(2)*expPar(end-1)+GP_hyperHyper(1))^3);
                
                GradGradGradLogPrior_GPhyperPar(2) = -(expPar(end)*(1-expPar(end)))/...
                    ((1+expPar(end))^3) - ...
                    ( GP_hyperHyper(3)*GP_hyperHyper(4)*expPar(end)*...
                    (GP_hyperHyper(3)-GP_hyperHyper(4)*expPar(end)) )/...
                    ((GP_hyperHyper(4)*expPar(end)+GP_hyperHyper(3))^3);
                
                %GradGradGradLogPrior_GPhyperPar(3) = 0;
                
                %GradGradGradLogPrior_GPhyperPar(3) = GP_hyperHyper(6)/exp(param_sc(7));
                
            end % grad23_EmInd(2)
            
        end % em_ind && grad23_EmInd(1)
        
    end % gp_ind
    
    LogPrior = LogPrior_bioPar + LogPrior_GPhyperPar;
    
    if em_ind == 0 && grad1_SimInd == 0
        GradLogPrior = NaN(nd,1);
    else
        GradLogPrior = [GradLogPrior_bioPar, GradLogPrior_GPhyperPar];
        GradLogPrior = GradLogPrior';
    end
    
    if em_ind == 1 && grad23_EmInd(1) == 1
        GradGradLogPrior = [GradGradLogPrior_bioPar; GradGradLogPrior_GPhyperPar];
        GradGradLogPrior = GradGradLogPrior';
        %
        if grad23_EmInd(2) == 1
            GradGradGradLogPrior = [GradGradGradLogPrior_bioPar; GradGradGradLogPrior_GPhyperPar];
            GradGradGradLogPrior = GradGradGradLogPrior';
            
        end % grad23_EmInd(2)
        
    end %em_ind && grad23_EmInd(1)
    
    
end % corrErr

% Return the log posterior and its gradient
%LogLik = 0; GradLogLik = 0;
%LogPrior = 0; GradLogPrior = 0;

if pass == 1
    LogPosterior = LogLik + LogPrior;
    
    GradLogPosterior = GradLogLik + GradLogPrior;
    
    if em_ind == 1 && grad23_EmInd(1) == 1 && phase_ind == 2
        
        GradGradLogPosterior = GradGradLogLik + GradGradLogPrior;
        
        if grad23_EmInd(2) == 1
            
            GradGradGradLogPosterior = GradGradGradLogLik + GradGradGradLogPrior;
            
        else
            
            GradGradGradLogPosterior = NaN(nd,1);
            
        end % grad23_EmInd(2)
        
    else
        % NaNs if simulator used or we only need 1st gradient for the
        % emulator (i.e use HMC or NUTS, not RMHMC, BAHMC or LMC)
        % or phase_ind = 1, for which we do not need 2nd and 3rd derivative
        
        GradGradLogPosterior = NaN(nd,1);
        
        GradGradGradLogPosterior = NaN(nd,1);
        
    end
    
else % pass = 0
    
    LogPosterior = -10^10; % equivalent to Posterior = 0
    GradLogPosterior = zeros(nd,1);
    if em_ind == 0
        GradGradLogPosterior = zeros(nd,1);
        GradGradGradLogPosterior = zeros(nd,1);
    end
    
    if grad23_EmInd(1) == 1
        GradGradLogPosterior = zeros(nd,1);
        if grad23_EmInd(2) == 1
            GradGradGradLogPosterior = zeros(nd,1);
        else
            
            GradGradGradLogPosterior = NaN(nd,1);
            
        end % grad23_EmInd(2)
        
    else
        % NaNs if simulator used or we only need 1st gradient for the
        % emulator (i.e use HMC or NUTS, not RMHMC, BAHMC or LMC)
        % or phase_ind = 1, for which we do not need 2nd and 3rd derivative
        
        GradGradLogPosterior = NaN(nd,1);
        
        GradGradGradLogPosterior = NaN(nd,1);
        
    end % grad23_EmInd(1)
    
end % pass

end