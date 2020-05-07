function [current_p,LogPosterior_sim,LogPosterior_em,GradLogPost_em,...
    ObjFct_sim, gp_regr, x_regr, y_regr, mean_y, std_y] = ...
    HMC_pulm_MoreEfficient(current_p, sigma2, epsilon, L, ...
    gp_regr, x_regr, y_regr, gp_class, x_class, y_class, ...
    nd, phase_ind, truePressure, ...
    alp, bet, GP_hyperHyper, extra_p, l, u, sc, ...
    LogPosterior_sim_begin, LogPosterior_em_begin, ...
    GradLogPost_em_begin, ObjFct_sim_begin, ...
    mean_y, std_y, do_nuts, corrErr, gp_ind, M, do_DA)
% Generic function for HMC in the pulmonary problem
% p: position has to be the unbounded variable, param_sc
% epsilon: step size
% q: momentum
% phase_ind will be fixed input argument, em_ind will change within this
% function

p = current_p;

% Update the momentum variable, q ~ MVN(0,I)
q = mvnrnd(zeros(nd,1),eye(nd)); %!!!!!! eye(nd)
current_q = q;

% Obtain LogPost and GradLogPost at the beginning of the trajectory
% It may not be the same one as that saved at the previous iteration for
% the accepted point since the noise variance, if sampled in a Gibbs step
% will lead to a different LogLik

if corrErr == 0 && phase_ind == 2
    
    em_ind = 1; % use emulator
    grad1_SimInd = NaN; grad23_EmInd = [0 0];
    [LogPosterior_em_begin,GradLogPost_em_begin] = ...
        HMCDerivPosterior_all_MoreEfficient_Generic(p, ...
        sigma2, truePressure, alp, bet, GP_hyperHyper, extra_p, ...
        l, u, sc, em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
        gp_regr, x_regr, y_regr, gp_class, x_class, y_class, ...
        mean_y, std_y, do_nuts, corrErr, gp_ind);
    
    n = numel(truePressure);
    
    LogPosterior_sim_begin = -n/2*log(sigma2)-n/2*log(2*pi) - ...
        ObjFct_sim_begin/(2*sigma2) + ...
        Prior_Log_GradLog(p, alp, bet, do_nuts);
    
end % corrErr

% Evaluate potential and kinetic energy at start of trajectory
current_E_pot = - LogPosterior_em_begin;
current_E_kin = 0.5 * current_q/M*current_q';

current_H = current_E_pot + current_E_kin;

% Use the gradient of the log posterior density of p to make half step of q
q = q + 0.5 * epsilon * GradLogPost_em_begin';

% Leapfrog scheme: alternate full steps for position and momentum

for i = 1:L % L: no of steps
    
    % Make a full step for the position
    p = p + epsilon * (M\q')';
    
    % Make a full step for the momentum, except at end of trajectory
    if i~=L
        em_ind = 1; % use emulator within the trajectory
        grad1_SimInd = NaN; grad23_EmInd = [0 0];
        [~, GradLogPost_trj, ~, ~, Var, ~] = ...
            HMCDerivPosterior_all_MoreEfficient_Generic(p, ...
            sigma2, truePressure, alp, bet, GP_hyperHyper, extra_p, ...
            l, u, sc, em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
            gp_regr, x_regr, y_regr, gp_class, x_class, y_class, ...
            mean_y, std_y, do_nuts, corrErr, gp_ind);
        
        if phase_ind == 1 % exploratory
            % if sqrt(Var) >= 3, stop simulation and evaluate expensive target
            % density and its gradient at the point where we stopped
            
            if sqrt(Var) >= 3
                disp('var too high')
                Var
                break
            end
        end
        
        q = q + epsilon * GradLogPost_trj';

    end
end

if phase_ind == 1 && sqrt(Var) >= 3
    
    if corrErr == 1
        
        if gp_ind ~= 5
            
            param = [(u(1:end-2).*exp(p(1:end-2))+l(1:end-2))./(1+exp(p(1:end-2))) ...
                exp(p(end-1:end))]; % parameters on original scale
            
        else
            
            param = (u.*exp(p)+l)./(1+exp(p)); % parameters on original scale
            
        end
        
    else % corrErr = 0
        
        param = (u.*exp(p)+l)./(1+exp(p)); % parameters on original scale
        
    end
    
    param_em = param./sc; % parameters used in emulation
    
    [ObjFct, pass] = Run_simulator(param_em, extra_p, ...
        truePressure, sc, gp_ind, corrErr);
    
    x_regr(end+1,:) = param_em;
    y_regr = y_regr .* std_y + mean_y; % bring on original scale
    y_regr(end+1) = ObjFct; % original scale
    
    mean_y = mean(y_regr);
    std_y = std(y_regr);
    y_regr = (y_regr - mean_y)./std_y;
    
    gp_regr = ReFitGP(x_regr, y_regr);
    
    % repeat process for pass with gp_class if need be
    % ....
    
end % sqrt(Var) >= 3

% Make a half step for momentum at the end
em_ind = 1; % use emulator within the trajectory
grad1_SimInd = NaN; grad23_EmInd = [0 0];
[LogPosterior_em_end, GradLogPost_em_end] = ...
    HMCDerivPosterior_all_MoreEfficient_Generic(p, ...
    sigma2, truePressure, alp, bet, GP_hyperHyper, extra_p, l, u, sc, ...
    em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
    gp_regr, x_regr, y_regr, gp_class, x_class, y_class, mean_y, std_y, ...
    do_nuts, corrErr, gp_ind);

q = q + 0.5 * epsilon * GradLogPost_em_end';

% Negate momentum at end of trajectory to make the proposal symmetric
q = -q;

% Evaluate potential and kinetic energy at end of trajectory
% H = E_pot + E_kin; E_pot = -LogPost; E_kin = 0.5*sum(q.^2);
% joint posterior distr: p(p,q) = exp(-H)
proposed_E_pot = - LogPosterior_em_end;
proposed_E_kin = 0.5 * q/M*q';
proposed_H = proposed_E_pot + proposed_E_kin;

if do_DA == 0 % no delayed acceptance
    
    % Accept or reject the state at end of trajectory
    % (in a Metropolis step, with proposal distribution coming from the hamiltonian dynamics on emulated space,
    % returning either the position p at end of trajectory or the initial position
    
    % current-proposed and not proposed-current as it would normally because
    % because proposed is actually -logpost
    
    em_ind = 0; % use simulator at end of trajectory
    grad1_SimInd = 0; grad23_EmInd = [NaN NaN];
    [LogPosterior_sim_end, ~, ~, ~, ~, ObjFct_sim_end] = ...
        HMCDerivPosterior_all_MoreEfficient_Generic(p, ...
        sigma2, truePressure, alp, bet, GP_hyperHyper, ...
        extra_p, l, u, sc, em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
        gp_regr, x_regr, y_regr, gp_class, x_class, y_class, mean_y, std_y, ...
        do_nuts, corrErr, gp_ind);
    
    r = - LogPosterior_sim_begin + LogPosterior_sim_end + current_E_kin - proposed_E_kin;
    
    [current_E_pot, proposed_E_pot]
    [current_E_kin, proposed_E_kin]
    
    if r > 0 || (r > log(rand)) % accept
        disp('accept')
        LogPosterior_sim = LogPosterior_sim_end;
        LogPosterior_em = LogPosterior_em_end;
        GradLogPost_em = GradLogPost_em_end;
        ObjFct_sim = ObjFct_sim_end;
        current_p = p;
        
    else % reject
        disp('reject')
        LogPosterior_sim = LogPosterior_sim_begin;
        LogPosterior_em = LogPosterior_em_begin;
        GradLogPost_em = GradLogPost_em_begin;
        ObjFct_sim = ObjFct_sim_begin;
        current_p = current_p;
        
    end
    
else % do_DA = 1 -- do delayed acceptance
    
    if proposed_E_pot == 10^10
        disp('reject in stage 1 because of zero likelihood')
        LogPosterior_sim = LogPosterior_sim_begin;
        LogPosterior_em = LogPosterior_em_begin;
        GradLogPost_em = GradLogPost_em_begin;
        ObjFct_sim = ObjFct_sim_begin;
        current_p = current_p;
        
    else
        
        r1 = -proposed_H + current_H;
        
        if r1 > 0 || (r1 > log(rand)) % accept in stage 1
            
            disp('accept in stage 1')
            
            % next calculate acc rate in stage 2 using the simulator in an MH step
            em_ind = 0;
            grad1_SimInd = 0; grad23_EmInd = [NaN, NaN];
            [LogPosterior_sim_end, ~, ~, ~, ~, ObjFct_sim_end] = ...
                HMCDerivPosterior_all_MoreEfficient_Generic(p, ...
                sigma2, truePressure, alp, bet, GP_hyperHyper, ...
                extra_p, l, u, sc, em_ind, phase_ind, grad1_SimInd, grad23_EmInd, ...
                gp_regr, x_regr, y_regr, gp_class, x_class, y_class, mean_y, std_y, ...
                do_nuts, corrErr, gp_ind);
            
            [LogPosterior_sim_begin,LogPosterior_em_begin]
            [LogPosterior_sim_end,LogPosterior_em_end]
            
            [current_E_pot, proposed_E_pot];
            [current_E_kin, proposed_E_kin];
            
            %r2 = LogPosterior_sim_end - LogPosterior_sim_begin - current_H + proposed_H;
            r2 = LogPosterior_sim_end - LogPosterior_sim_begin + ...
                LogPosterior_em_begin - LogPosterior_em_end;
            
            if r2 > 0 || (r2 > log(rand)) % accept at 2nd stage
                disp('accept in stage 2')
                LogPosterior_sim = LogPosterior_sim_end;
                LogPosterior_em = LogPosterior_em_end;
                GradLogPost_em = GradLogPost_em_end;
                ObjFct_sim = ObjFct_sim_end;
                current_p = p;
                
            else % reject at 2nd stage
                disp('reject in stage 2')
                LogPosterior_sim = LogPosterior_sim_begin;
                LogPosterior_em = LogPosterior_em_begin;
                GradLogPost_em = GradLogPost_em_begin;
                ObjFct_sim = ObjFct_sim_begin;
                current_p = current_p;
                
            end % r2
            
        else % r1
            disp('reject in stage 1')
            LogPosterior_sim = LogPosterior_sim_begin;
            LogPosterior_em = LogPosterior_em_begin;
            GradLogPost_em = GradLogPost_em_begin;
            ObjFct_sim = ObjFct_sim_begin;
            current_p = current_p;
            
        end % r1
        
    end % proposed_E_pot
    
end % do_DA

end

