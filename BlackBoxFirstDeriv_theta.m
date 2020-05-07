function FirstDeriv = BlackBoxFirstDeriv_theta(pressure, param, extra_p)

% Calculates the derivative of the black-box sor06 w.r.t. every parameter 

% input: pressure, param, m, k, taper, verbosity, numHeartBeats, id

% output: the first derivative for every parameter

sor0 = pressure;

id = extra_p(1); nd = extra_p(2); HB = extra_p(3); cycles = extra_p(4);

FirstDeriv = NaN(nd, numel(sor0));

parfor i = 1:nd
    
    if i == 1 %f3
        h = 0.005;%0.05;
        param_hplus = [param(1)+h, param(2), param(3), param(4), param(5)];
		
	elseif i == 2 % m1
        h = 0.0005;
        param_hplus = [param(1), param(2)+h, param(3), param(4), param(5)];
        
    elseif i == 3 % rr1
        h = 0.0005;
        param_hplus = [param(1), param(2), param(3)+h, param(4), param(5)];
        
    elseif i == 4 % rr2
        h = 0.00001; %0.005;
        param_hplus = [param(1), param(2), param(3), param(4)+h, param(5)];
        
    else % cc1
        h = 0.00003; %0.00005; %0.0003;
        param_hplus = [param(1), param(2), param(3), param(4), param(5)+h];
        
    end
    
    cx = unix(sprintf('./sor06 %f %f %f %f %f %f %f %d', ...
     param_hplus(1), param_hplus(2), param_hplus(3), param_hplus(4), param_hplus(5), HB, cycles, id+i*100));
    
    if cx == 0
        state = CreateData_Optim(id+i*100);
        sorh_plus = state(end/2+1:end);
    else
        sorh_plus = repmat(0, numel(sor0), 1);
    end
    
    FirstDeriv(i,:) = (sorh_plus - sor0)./h;
    
end


end
