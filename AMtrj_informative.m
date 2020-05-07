function [current_p, ObjFct] = AMtrj_informative(current_p, sigma2, R, em_int, extra_p, ...
    gp_regr, x_regr, y_regr, ...
    gp_class, x_class, y_class, ...
    nd, truePressure, l, u, sc, phase_ind, ...
    ObjFct, extraPar_gp, gp_ind, corrErr, alp, bet, GPhyper)

oldpar = current_p./sc; % we will use oldpar in emulation
chain_em = NaN(em_int,nd);
chain_em(1,:) = oldpar;
current_sim_ObjFct = ObjFct;

oldprior = Prior_AM(oldpar,sc,l,u,gp_ind,alp,bet,GPhyper,corrErr);

acc = 0; rejout = 0;

em_ind = 1; % we will emulate ss or loglik within the trajectory, as follows:
oldObjFct_em = mice_pulm_ss(oldpar,truePressure,...
    gp_regr, x_regr, y_regr, gp_class, x_class, y_class, ...
    extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr);

for i = 2:em_int
    % sample new candidate from Gaussian proposal
    q = randn(1,nd);
    newpar = oldpar + q*R;
    
    if ( any(newpar.*sc<l) || any(newpar.*sc>u) && ( corrErr == 0 || (corrErr == 1 && gp_ind == 5) ) ) ||...
   ( any(newpar(1:end-2).*sc(1:end-2)<l(1:end-2)) || any(newpar(1:end-2).*sc(1:end-2)>u(1:end-2)) && (corrErr == 1 && gp_ind ~= 5) )
        % original points outside boundaries - amplitude and lengthscale not within hard bounds
        %newpar.*sc
        if corrErr == 0
            newObjFct_em = 10^10; %ss
        else
            newObjFct_em = -10^10; %loglik
            
            newprior = 0;
        end
        
        rejout = rejout + 1;
        
    else % inside the boundaries
        
        newObjFct_em = mice_pulm_ss(newpar,truePressure,...
            gp_regr, x_regr, y_regr, gp_class, x_class, y_class, ...
            extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr);
        
        newprior = Prior_AM(newpar,sc,l,u,gp_ind,alp,bet,GPhyper,corrErr);
        
    end % inside/outside boundaries
    
    if corrErr == 0
        if newObjFct_em == 10^10 || newObjFct_em < 0 % unsuccessful simulation (classifier) or outside boundaries
            tst = 0;
        else
            tst = exp( -0.5/sigma2*(newObjFct_em-oldObjFct_em) + newprior-oldprior );
        end
        
    else % corrErr = 1
        if newObjFct_em == -10^10 % unsuccessful simulation (classifier) or outside boundaries
            tst = 0;
        else
            tst = exp(newObjFct_em - oldObjFct_em + newprior-oldprior);
        end
    end % corrErr=1
    
    if tst <= 0
        accept = 0;
    elseif tst >= 1
        accept = 1; acc = acc + 1;
    elseif tst > rand(1,1)
        accept = 1; acc = acc + 1;
    else
        accept = 0;
    end
    
    if accept == 1 % accept proposal
        
        chain_em(i,:) = newpar;
        
        oldpar = newpar;
        
        oldObjFct_em = newObjFct_em;
        
        oldprior = newprior;
        
    else % reject
        
        chain_em(i,:) = oldpar;
        
        oldpar = oldpar;
        
        oldObjFct_em = oldObjFct_em;
        
        oldprior = oldprior;
        
    end
    
end

acc/em_int;
rejout/em_int;

if rejout > em_int/2
    disp('Too many proposals outside boundaries, so decrease cov_MH entries')
end

em_ind = 1;
oldObjFct_em = mice_pulm_ss(chain_em(1,:), truePressure,...
    gp_regr, x_regr, y_regr, gp_class, x_class, y_class,...
    extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr);
newObjFct_em = mice_pulm_ss(chain_em(end,:), truePressure,...
    gp_regr, x_regr, y_regr, gp_class, x_class, y_class,...
    extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr);

% Assess using simulator whether we accept or reject the newly proposed point
if oldObjFct_em == newObjFct_em
    disp('emulator rejects') % the emulator hasn't accepted any points along the trajectory
    accept = 0; current_p = current_p; ObjFct = current_sim_ObjFct;
else
    em_ind = 0; % use simulator at end of trajectory, i.e. for chain_em(end).*sc
    p = chain_em(end,:).*sc;
    chain_em(end,1:4).*sc(1:4)
    %chain_em(end,5:6).*sc(5:6)
    
    proposed_sim_ObjFct = mice_pulm_ss(p, truePressure,...
        gp_regr, x_regr, y_regr, gp_class, x_class, y_class,...
        extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr);
    
    [proposed_sim_ObjFct, newObjFct_em]; % compares proposed according to simulator and to emulator
    
    oldprior = Prior_AM(chain_em(1,:),sc,l,u,gp_ind,alp,bet,GPhyper,corrErr);
    newprior = Prior_AM(chain_em(end,:),sc,l,u,gp_ind,alp,bet,GPhyper,corrErr);
    
    if corrErr == 0
        % here sigma2 should come from sampling with the simulator
        tst = exp( -0.5/sigma2 * ( proposed_sim_ObjFct - current_sim_ObjFct + ...
            oldObjFct_em - newObjFct_em ) );
        proposed_sim_ObjFct - newObjFct_em
        -current_sim_ObjFct + oldObjFct_em
    else
        tst = exp(proposed_sim_ObjFct-current_sim_ObjFct + ...
            oldObjFct_em - newObjFct_em); % exact priors cancel out from the em and sim
        proposed_sim_ObjFct - newObjFct_em
        -current_sim_ObjFct + oldObjFct_em
    end
    
    if tst <= 0
        disp('reject in step2')
        accept = 0; current_p = current_p; ObjFct = current_sim_ObjFct;
    elseif tst >= 1
        disp('accept in step2')
        accept = 1; current_p = p; ObjFct = proposed_sim_ObjFct;
    elseif tst > rand(1,1)
        disp('accept in step2')
        accept = 1; current_p = p; ObjFct = proposed_sim_ObjFct;
    else
        disp('reject in step2')
        accept = 0; current_p = current_p; ObjFct = current_sim_ObjFct;
    end
end

end