function prior = Prior_AM(par,sc,l,u,gp_ind,alp,bet,GPhyper,corrErr)

if corrErr == 1
    
    priorBio = sum( (alp-1) .* log(par(1:end-2).*sc(1:end-2)-l(1:end-2)) + ...
    (bet-1) .* log(u(1:end-2) - par(1:end-2).*sc(1:end-2)) );

    if gp_ind ~= 5
        priorCov = -log(par(end-1)*sc(end-1)) - log(sqrt(GPhyper(2)*2*pi)) - ...
            (log(par(end-1)*sc(end-1)) - GPhyper(1))^2/(2*GPhyper(2)) + ...
            -log(par(end)*sc(end)) - log(sqrt(GPhyper(4)*2*pi)) - ...
            (log(par(end)*sc(end)) - GPhyper(3))^2/(2*GPhyper(4));
    else
        priorCov = -log(par(end-1)*sc(end-1)) - log(GPhyper(2)-GPhyper(1)) + ...
            -log(par(end)*sc(end)) - log(GPhyper(4)-GPhyper(3));
        
    end
    
    prior = sum(log(sc)) + priorBio + priorCov;
    
else %corrErr
    
    priorBio = sum ((alp-1) .* log(par.*sc-l) + (bet-1) .* log(u - par.*sc) );

    prior = sum(log(sc)) + priorBio;
    
end

