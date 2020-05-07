function [ObjFct, pass, Z, C, L, pressure] = Run_simulator(x, extra_p, ...
    truePressure, sc, gp_ind, corrErr)

ntp = size(truePressure,1); % number of time points

if corrErr == 0
    
    f3_vec = x(:,1) * sc(1);
    m1_vec = x(:,2) * sc(2);
    rr1_vec = x(:,3) * sc(3);
    rr2_vec = x(:,4) * sc(4);
    cc1_vec = x(:,5) * sc(5);
    
    n = size(x,1);
    
    id = extra_p(1); HB = extra_p(3); cycles = extra_p(4);
    
    %truePressure = trueState(end/2+1:end);
    
    pass = NaN(n,1);
    NLL = NaN(n,1); Z = NaN(ntp,n);
    
    for i = 1:n
        % 98463.12095 1.74987 0.18561 -1.46490
        param = [f3_vec(i), rr1_vec(i), rr2_vec(i), cc1_vec(i)];
        cx = unix(sprintf('./sor06 %f %f %f %f %f %f %f %d', ...
            param(1), param(2), param(3), param(4), param(5), HB, cycles, id));
        
        if cx == 0
            
            pass(i) = 1;
            
            state = CreateData_Optim(id);
            pressure = state(end/2+1:end);
            
            NLL(i) = sum((pressure-truePressure).^2);
            Z(:,i) = truePressure - pressure;
            
        else
            
            pass(i) = 0;
            NLL(i) = 1e+10; % very large value for unsuccessful simulation
            
        end
    end
    
    ObjFct = NLL; C = NaN; L = NaN;
    
else % corrErr = 1
    
    f3_vec = x(:,1) * sc(1);
    m1_vec = x(:,2) * sc(2);
    rr1_vec = x(:,3) * sc(3);
    rr2_vec = x(:,4) * sc(4);
    cc1_vec = x(:,5) * sc(5);
    
    if gp_ind == 5 % neural network cov fct
        ws2_vec = x(:,6) * sc(6);
        bs2_vec = x(:,7) * sc(7);
    else % sw exp, matern 3/2, 5/2, periodic cov fct
        magns2_vec = x(:,6) * sc(6);
        lgt_vec = x(:,7) * sc(7);
    end
    
    n = size(x,1);
    
    id = extra_p(1); HB = extra_p(3); cycles = extra_p(4);
    
    
    pass = NaN(n,1);
    loglik = NaN(n,1);
    
    for i = 1:n
        
        param = [f3_vec(i), m1_vec(i), rr1_vec(i), rr2_vec(i), cc1_vec(i)];
        cx = unix(sprintf('./sor06 %f %f %f %f %f %f %f %d', ...
            param(1), param(2), param(3), param(4), param(5), HB, cycles, id));
        
        if cx == 0
            
            pass(i) = 1;
            
            state = CreateData_Optim(id);
            pressure = state(end/2+1:end);
            
            res = truePressure-pressure;
            
            T = 0.11;  % Cycle length from sor06.h
            deltaT = T/(ntp-1);
            t = 0:deltaT:T;
            
            t = t';
            
            y = (res-mean(res))./std(res);
                        
            if gp_ind == 1
                
                lik = lik_gaussian('sigma2', 2.82e-06, 'sigma2_prior', prior_fixed);

                gpcf = gpcf_sexp('lengthScale', lgt_vec(i), ...
                    'magnSigma2', magns2_vec(i) ,...
                    'lengthScale_prior', prior_fixed, 'magnSigma2_prior', prior_fixed);
                
            elseif gp_ind == 2
                %???
                lik = lik_gaussian('sigma2', 2.82e-06, 'sigma2_prior', prior_fixed); % replace 0 by optimum

                gpcf = gpcf_matern32('lengthScale', lgt_vec(i), ...
                    'magnSigma2', magns2_vec(i) ,...
                    'lengthScale_prior', prior_fixed, 'magnSigma2_prior', prior_fixed);
                
            elseif gp_ind == 3
                %???
                lik = lik_gaussian('sigma2', 2.82e-06, 'sigma2_prior', prior_fixed); % replace 0 by optimum
                
                gpcf = gpcf_matern52('lengthScale', lgt_vec(i), ...
                    'magnSigma2', magns2_vec(i) ,...
                    'lengthScale_prior', prior_fixed, 'magnSigma2_prior', prior_fixed);
                
            elseif gp_ind == 4
                %???
                lik = lik_gaussian('sigma2', 2.82e-06, 'sigma2_prior', prior_fixed);

                gpcf = gpcf_periodic('magnSigma2',magns2_vec(i),...
                    'lengthScale',lgt_vec(i), ...
                    'period',3.5, 'decay',0,'lengthScale_sexp',0.008, ...
                    'lengthScale_prior',prior_fixed,'magnSigma2_prior',prior_fixed, ...
                    'lengthScale_sexp_prior',prior_fixed);
                
            else
                %???
                lik = lik_gaussian('sigma2', 2.44e-06, 'sigma2_prior', prior_fixed);

                gpcf = gpcf_neuralnetwork('biasSigma2', bs2_vec(i), ...
                    'weightSigma2', ws2_vec(i), ...
                    'biasSigma2_prior',prior_fixed, ...
                    'weightSigma2_prior',prior_fixed);
            end
            
            gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-6);
            
            [~,C] = gp_trcov(gp, t);
            
            [L,posdef] = chol(C,'lower');
            if posdef == 0
                a = (L\(pressure-truePressure));
                MH = a'*a; % Mahalanobis distance
                LogDet = ntp * log(2*pi) + 2*sum(log(diag(L)));
                loglik(i) = -0.5*LogDet - 0.5*MH;
                Z = res;
            else
                pass(i) = 0;
                loglik(i) = -10^10; % likelihood zero
                Z = repmat(-10^10,ntp,1); 
                C = NaN(ntp,ntp);  
                L = NaN(ntp,ntp);
            end
            
        else
            
            pass(i) = 0;
            loglik(i) = -10^10; % likelihood zero
            pressure = NaN(ntp,1);
            Z = repmat(-10^10,ntp,1); 
            C = NaN(ntp,ntp);  
            L = NaN(ntp,ntp);
        end
    end
    
    ObjFct = loglik;
    
end

end

