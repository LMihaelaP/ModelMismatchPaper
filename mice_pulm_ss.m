function ObjFct = mice_pulm_ss(param,truePressure,...
    gp_regr, x_regr, y_regr, ...
    gp_class, x_class, y_class, ...
    extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr)
% pulmonary sum-of-squares or Mahalanobis distance function
% param is on emulation scale when em_ind = 1 and on original scale when
% em_ind = 0.


if em_ind == 1 % within emulation interval
    
    mean_y = extraPar_gp(1); std_y = extraPar_gp(2);
    
    [E,Var] = gp_pred(gp_regr, x_regr, y_regr, param);
    
    ObjFct = E * std_y + mean_y;
    
    [~, ~, lpyt_la, ~, ~] = gp_pred(gp_class, x_class, y_class, param, 'yt', 1);
    
    if exp(lpyt_la) < 0.8 % unsuccessful simulation
        
        if corrErr == 0 % rss
            
            ObjFct = 10^10;
            
        else % loglik
            
            ObjFct = -10^10;
            
        end
        
    end
    
else % outside emulation interval, so evaluate PDEs
    
    ObjFct = Run_simulator(param./sc, extra_p, truePressure, sc, ...
        gp_ind, corrErr);
    
    
end


end
