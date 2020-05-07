%Construct an emulator for the log likelihood 
clear; close all;

% Add path to GPstuff toolbox
GPpath = genpath('GPstuff-4.7'); % path including GPstuff & all folders below it
addpath(GPpath); % add the folder and its subfolders to the search path
addpath('MCMC_run')

%% Load the real data
hypo = 0; % indicator if hypoxic mouse (1) or control mouse (0) is used

if hypo
   trueFlow = importdata('qH1_512.dat');
   truePressure = importdata('pH1_512.dat');
else
   trueFlow = importdata('qC6_512.dat');
   truePressure = importdata('pC6_512.dat');
end

trueState = [trueFlow; truePressure];

nd = 7; % no of parameters
HB = 12; % no of heartbeats
cycles = 1;
ntp = size(truePressure,1);
% Type of covariance function used
% 1 - squared exponential; 2 - matern 3/2; 3 - matern 5/2; 
% 4 - periodic; 5 - neural network
gp_ind = 5;

id = 1000*gp_ind; % id for data files
extra_p = [id, nd, HB, cycles];

if gp_ind ~= 5
    % Bounds for original parameters
    l = [3e04, 1, 0.05, 0.05, 0.05, 0.5, 0.001];%3.4663e04
    u = [5e05, 2*pi, 2.5,  2.5,  2.5, 1.5,  0.06];%8.6659e04
    % Parameter scaling for emulation
    sc = [10^6, 10, 10, 10, 10, 10, 10^(-1)];
    
else    
    % Bounds for original parameters
    l = [3e04, 1, 0.05, 0.05, 0.05, 10^4, 1];
    u = [5e05, 2*pi, 2.5, 2.5, 2.5, 90000,   500];
    sc = [10^6, 10, 10, 10, 10, 10^6, 10^3];
end
    

%% Define prior (for original, unscaled parameters) 
% Derived from rescaled beta distribution on the bounded biological parameters theta
% Be(1,1) = Uniform
alp = [1,1,1,1,1]; 
bet = [1,1,1,1,1];
% For GP hyperparameters
% for sq exp, matern 3/2, 5/2, periodic:
% amplitude ~ LogGauss(GPhyperHyper(1), GPhyperHyper(2)),LogGauss(mua,sigmaa^2)
% lengthscale ~ LogGauss(GPhyperHyper(3), GPhyperHyper(4)),LogGauss(mul,sigmal^2)
% cov noise ~ InvGamma(GPhyperHyper(5), GPhyperHyper(6)),InvGamma(alpha, beta)
% for neural network:
% bias sigma2 ~ Logunif(GPhyperHyper(1), GPhyperHyper(2)),Logunif(loglb, logub) 
% weight sigma2 ~ Logunif(GPhyperHyper(3), GPhyperHyper(4)),Logunif(loglw, loguw) 
% cov noise ~ InvGamma(GPhyperHyper(5), GPhyperHyper(6)),InvGamma(alpha, beta)
if gp_ind ~= 5   
    GP_hyperHyper = [log(0.1), 0.095, log(0.0275), 0.1]; % amplitude:log(var(res))
else
    GP_hyperHyper = [l(nd-1), u(nd-1), l(nd), u(nd)];
end
%% Construct GP model out of D = 100 simulator callings
X = sobolset(7, 'Skip',1.4e4,'Leap',0.6e13); % draw 100 points

f3_vec = l(1) + (u(1)-l(1)) * X(:,1);
m1_vec = l(2) + (u(2)-l(2)) * X(:,2);
rr1_vec = l(3) + (u(3)-l(3)) * X(:,3);
rr2_vec = l(4) + (u(4)-l(4)) * X(:,4);
cc1_vec = l(5) + (u(5)-l(5)) * X(:,5);
GPhyper1_vec = l(6) + (u(6)-l(6)) * X(:,6);
GPhyper2_vec = l(7) + (u(7)-l(7)) * X(:,7);

par = [f3_vec, m1_vec, rr1_vec, rr2_vec, cc1_vec, GPhyper1_vec, GPhyper2_vec]./sc;

corrErr = 1; % we have correlated errors

% Run simulator to obtain log lik for training points
[loglik, pass] = Run_simulator(par, extra_p, truePressure, sc, ...
    gp_ind, corrErr);

pass = logical(pass);


%
%%%% INSTEAD OF FINDING THRESHOLD 'BY HAND'
n = size(loglik,1); k = 500; % no of points we want to keep to fit the GP regression
perc = 1-k/n;% keep the (1-k/n)th percentile (the largest k points)
T_ll = quantile(loglik,perc);
%%%

%%% OR
k = 500; % no of points we want to keep to fit the GP regression
temp = sort(loglik, 'descend');
T_ll = temp(k); % the threshold is the lowest loglik value out of the k values we keep to fit the GP regression
%%%

I_ll = loglik>T_ll;

% Construct classifier
x_class = par(I_ll,:); y_class = 2.*pass(I_ll)-1;
% For GP regr only look successful simulations from simulator
x_regr = par(pass&I_ll,:); y_regr = loglik(pass&I_ll);

%%%

mean_y = mean(y_regr);
std_y = std(y_regr);

loglik_max = max(max(loglik(loglik~=-10^10)));
loglik_min = min(min(loglik(loglik~=-10^10)));

y_regr = (y_regr-mean_y)./std_y; % mean 0 and std 1 of of y

% Build GP model (loglik emulator and classifier)
X_r = sobolset(9, 'Skip',2e12,'Leap',0.9e15); % 9 draw 10 values
n_r = size(X_r, 1);

l_r = [0.1  0.1  0.1  0.1  0.1  0.1 0.1 0.1 0.000000001];
u_r = [1 1 1 1 1 1 1 1 1];

% H_r matrix with magnsigma2, every lengthscale and sigma2 on separate rows
H_r = [l_r(1) + (u_r(1)-l_r(1)) * X_r(:,1), ...
    l_r(2) + (u_r(2)-l_r(2)) * X_r(:,2), ...
    l_r(3) + (u_r(3)-l_r(3)) * X_r(:,3), ...
    l_r(4) + (u_r(4)-l_r(4)) * X_r(:,4), ...
    l_r(5) + (u_r(5)-l_r(5)) * X_r(:,5), ...
    l_r(6) + (u_r(6)-l_r(6)) * X_r(:,6), ...
    l_r(7) + (u_r(7)-l_r(7)) * X_r(:,7), ...
    l_r(8) + (u_r(8)-l_r(8)) * X_r(:,8), ...
    l_r(9) + (u_r(9)-l_r(9)) * X_r(:,9)];
    
X_c = sobolset(8, 'Skip',4e8,'Leap',0.9e15); % 8 draw 10 values
n_c = size(X_c, 1);

l_c = [1 1   0.1 0.5  0.5 0.5 0.5 0.5];
u_c = [10 20   5   5  5 5 5 5];
% H_c matrix with magnsigma2, every lengthscale and sigma2 on separate rows
H_c = [l_c(1) + (u_c(1)-l_c(1)) * X_c(:,1), ...
    l_c(2) + (u_c(2)-l_c(2)) * X_c(:,2), ...
    l_c(3) + (u_c(3)-l_c(3)) * X_c(:,3), ...
    l_c(4) + (u_c(4)-l_c(4)) * X_c(:,4), ...
    l_c(5) + (u_c(5)-l_c(5)) * X_c(:,5), ...
    l_c(6) + (u_c(6)-l_c(6)) * X_c(:,6), ...
    l_c(7) + (u_c(7)-l_c(7)) * X_c(:,7), ...
    l_c(8) + (u_c(8)-l_c(8)) * X_c(:,8)]; 
H(1)=1;H(end+1:end+7)=0.01*[1 1 1 1 1 1 1];H(end+1)=exp(-9.950929526405522);
[gp_regr, nlml_regr, gp_class, nlml_class] = ...
    GPmodel(x_regr, y_regr, x_class, y_class, H_r,...
    H_c(1,:), 2, corrErr);

[w,s] = gp_pak(gp_regr);
disp(exp(w))

[w,s] = gp_pak(gp_class);
disp(exp(w))

%Make predictions using gp_regr
[E, Var] = gp_pred(gp_regr, x_regr, y_regr, x_regr);
figure(1); clf(1); plot(y_regr, '.'); 
hold on; plot(y_regr,y_regr,'-r')
xlabel('Train data'); ylabel('Predictions')

% Make predictions using gp_class
[Eft_la, Varft_la, lpyt_la, Eyt_la, Varyt_la] = ...
    gp_pred(gp_class, x_class, y_class, x_class, ...
    'yt', ones(size(x_class,1),1) );
figure(2); clf(2)
% if good classifier, 2 dots on the plot: one at (0,-1) & another at (1,1)
plot(exp(lpyt_la), y_class, '.', 'markersize', 20) 
xlabel('Predictions'); ylabel('Train labels')

save('GPHMC_NN_initial_corrErr_RealData_stiffWider.mat')

%% Exploratory phase
load('GPHMC_NN_initial_corrErr_RealData_stiffWider.mat')

phase_ind = 1; % phase index 1 (exploratory) in Rasmussen's algorithm

do_DA = 1; % do delayed acceptance

do_nuts = 0;

sigma2 = NaN;

x_regr_refitted = x_regr;
y_regr_refitted = y_regr;
gp_regr_refitted = gp_regr;

nSamples = 1500; % no of samples

M = eye(nd); % mass matrix for momentum

p = NaN(nSamples, nd); % parameter samples from the exploratory phase
ObjFct = NaN(nSamples, 1); % sum-of-square samples from the exploratory phase

% Initialise position vector
j = find(y_regr_refitted==max(y_regr_refitted));
p0 = x_regr_refitted(j,:) .* sc; % original scale

p(1,1:nd-2) = log((p0(1:nd-2)-l(1:nd-2))./(u(1:nd-2)-p0(1:nd-2))); % unbounded
if gp_ind ~= 5
    p(1,end-1:end) = log(p0(end-1:end));
else
    p(1,end-1:end) = log((p0(end-1:end)-l(end-1:end))./(u(end-1:end)-p0(end-1:end))); % unbounded
end

l_HMC = [0.0001,100]; u_HMC = [0.001,150];
epsilon = 0.0005;
L = 100;

em_ind = 0; % use simulator for beginning of trajectory
grad1_SimInd = 0; grad23_EmInd = [NaN NaN];
[LogPosterior_sim, ~, ~, ~, ~, ObjFct(1)] = ...
    HMCDerivPosterior_all_MoreEfficient_Generic(p(1,:), sigma2, truePressure, alp, bet, ...
    GP_hyperHyper,extra_p, l, u, sc, em_ind, phase_ind, ...
    grad1_SimInd, grad23_EmInd, ...
    gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
    gp_class, x_class, y_class, mean_y, std_y, do_nuts, corrErr, gp_ind);

em_ind = 1; % use simulator for beginning of trajectory
grad1_SimInd = 1; grad23_EmInd = [NaN NaN];
[LogPosterior_em, GradLogPost_em] = ...
    HMCDerivPosterior_all_MoreEfficient_Generic(p(1,:), sigma2, truePressure, alp, bet, ...
    GP_hyperHyper,extra_p, l, u, sc, em_ind, phase_ind, ...
    grad1_SimInd, grad23_EmInd, ...
    gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
    gp_class, x_class, y_class, mean_y, std_y, do_nuts, corrErr, gp_ind);

acc = 0;
j=1;

for k = 2:nSamples
    % for every i^th sample, run the trajectory
    % phase_ind = 1, so we run plain HMC, not DA HMC
    [p(k,:), LogPosterior_sim, LogPosterior_em, GradLogPost_em, ObjFct(k), ...
        gp_regr_refitted, x_regr_refitted, y_regr_refitted, mean_y, std_y] = ...
        HMC_pulm_MoreEfficient(p(k-1,:), sigma2, epsilon, L, ...
        gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
        gp_class, x_class, y_class, ...
        nd, phase_ind, truePressure, alp, bet, ...
        GP_hyperHyper, extra_p, l, u, sc, LogPosterior_sim, LogPosterior_em, ...
        GradLogPost_em, ObjFct(k-1), mean_y, std_y, ...
        do_nuts, corrErr, gp_ind, M, do_DA);
    
    % if new point accepted, at the end of every trajectory,
    % remove the j^th point and add the newly accepted point as a training
    % point in the GP, and re-fit the GP
    if all(p(k,:) ~= p(k-1,:)) % i.e. we've just accepted the new point
        acc = acc + 1;
        
        % starting from beginning, gradually remove the old [size(y_regr,1)] training
        % points whose rss > T_ss, as we accept new train points & refit GP
        if acc <= size(y_regr,1) % delete or skip when we've accepted
            % not i <= size(y_regr,1) bc i gets incremeneted even when not
            % accepting, so we will stop removing old training points too
            % fast, hence we will miss some
            % if acc stays < size(y_regr,1) we will miss removing some old
            % training points, so increase nSamples
            if (y_regr_refitted(j) * std_y + mean_y) < T_ll
                x_regr_refitted(j,:) = []; y_regr_refitted(j) = [];
            else
                j = j + 1; % skip deleting
            end
        end
        
        if gp_ind ~= 5
            param = [(u(1:end-2).*exp(p(k,1:end-2))+l(1:end-2))./(1+exp(p(k,1:end-2))) ...
                exp(p(k,end-1:end))]; % parameters on original scale
        else
            param = (u.*exp(p(k,:))+l)./(1+exp(p(k,:))); % parameters on original scale
        end
        
        param_em = param./sc; % parameters used in emulation
        
        if x_regr_refitted(end,:)~=param_em % i.e. if we haven't already
            % added this point as a consequence of sqrt(Var)>=3
            
            x_regr_refitted(end+1,:) = param_em;
            y_regr_refitted = y_regr_refitted .* std_y + mean_y; % bring on original scale
            y_regr_refitted(end+1) = ObjFct(k); % because rss is on original scale
            
            mean_y = mean(y_regr_refitted);
            std_y = std(y_regr_refitted);
            y_regr_refitted = (y_regr_refitted - mean_y)./std_y;
            
            gp_regr_refitted = gp_optim(gp_regr_refitted,x_regr_refitted,y_regr_refitted);
            
            [E, ~] = gp_pred(gp_regr_refitted, x_regr_refitted, y_regr_refitted, x_regr_refitted);
            [y_regr_refitted(1:10),E(1:10)]
            
        end
    end
    
    % Re-draw epsilon and L
    epsilon = l_HMC(1) + (u_HMC(1)-l_HMC(1))*rand;
    L = l_HMC(2) + (u_HMC(2)-l_HMC(2))*rand;
end
save('GPMCMC_NN_exploratory_RealData_widerStiff.mat')

y_regr_refitted = y_regr_refitted * std_y + mean_y;
k = 1300; % no of points we want to keep to fit the GP regression
temp = sort(y_regr_refitted, 'descend');
T_ll = temp(k);
I_ll = find(y_regr_refitted > T_ll);
y_regr_refitted = y_regr_refitted(I_ll);
mean_y = mean(y_regr_refitted);
std_y = std(y_regr_refitted);

y_regr_refitted = (y_regr_refitted - mean_y)./std_y;
x_regr_refitted = x_regr_refitted(I_ll,:);

% Refit GP with burnin phase removed
% Use this GP in the sampling phase
gp_regr_refitted = gp_optim(gp_regr_refitted,x_regr_refitted,y_regr_refitted);


%% Sampling phase with AM

load('GPMCMC_NN_exploratory_RealData_widerStiff.mat')

nrun = 10; % run 10 chains in parallel
nSamples = 5000; % no of sampling phase samples
nburnin = 100; % no of burnin phase samples
em_int = 50; % emulation interval
phase_ind = 2;
adapt_int = 10; % adaptation interval
scale = 1; % scale the original paramerts bc of varying mgnitudes
% proposal covariance for MH within the trajectory
cov_MH = diag(repmat(5*10^(-8),nd,1)); 
extraPar_gp = [mean_y, std_y];
corrErr = 1; % correlated errors

delete(gcp('nocreate'))
parpool('local', nrun)

par_sim = cell(nrun,1); % parameter samples from the sampling phase
ObjFct_sim = cell(nrun,1); % sum-of-square samples from the sampling phase

% Initialise parameter chain
for j = 1:nrun
    par_sim{j}(1,:) = x_regr_refitted(end-100*j+2,:) .* sc; % original scale j instead of end!!
end

sigma2 = NaN; 
S20 = NaN; % prior for sigma2
N0 = NaN; % prior accuracy for S20

parfor j = 1:nrun %parfor
    extra_p = [1000*gp_ind+j, nd, HB, cycles];
    em_ind = 0; % use simulator for beginning of trajectory
    ObjFct = mice_pulm_ss(par_sim{j}(1,:),truePressure,...
        gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
        gp_class, x_class, y_class, ...
        extraPar_gp, em_ind, phase_ind, extra_p, sc, gp_ind, corrErr);
    
    if ObjFct ~= 1
        ObjFct_sim{j}(1) = ObjFct;
    else
        disp('Choose different starting values for the parameters')
    end
end

qcov_adjust = 1e-8; % epsilon adjustment for chain covariance
qcov_scale = 2.4 / sqrt(nd); % s_d from recursive covariance formula
sample_sigma = NaN; % sample sigma2 only for uncorrelated errors
acc = zeros(nrun,1); % acceptance rate for every chain

% Store cpu times for every run and average the at the end
initime = NaN(nrun,1);
fintime = NaN(nrun,1);

% Run 10 chains in parallel from different initialisations and different
% random seed generators
% if gp_ind == 2
%     u(5) = 3;
% end
parfor j = 1:nrun %parfor

	initime(j) = cputime;
	
    extra_p = [1000*gp_ind+j, nd, HB, cycles];
    % covariance update uses these to store previous values
    covchain = []; meanchain = []; wsum = []; lasti = 0;
    R = chol(cov_MH);
for i=2:nSamples+nburnin
    % for every i^th sample, run the trajectory
    [par_sim{j}(i,:), ObjFct_sim{j}(i)] = ...
        AMtrj_informative(par_sim{j}(i-1,:), sigma2, R, em_int, extra_p, ...
        gp_regr_refitted, x_regr_refitted, y_regr_refitted, ...
        gp_class, x_class, y_class, ...
        nd, truePressure, l, u, sc, ntp, N0, S20, sample_sigma, phase_ind, ...
        ObjFct_sim{j}(i-1), extraPar_gp, gp_ind, corrErr, alp, bet, GP_hyperHyper);
    if all(par_sim{j}(i,:) ~= par_sim{j}(i-1,:)) % just accepted the new point
        acc(j) = acc(j) + 1;
        disp(sprintf('accept for run %d', j))
        par_sim{j}(i,:);
    end
    
    
    if mod(i, adapt_int) == 0 % we adapt
        disp('we adapt')
        if scale == 1 % calculate the chain covariance for the transformed parameters
          [covchain,meanchain,wsum] = covupd(par_sim{j}((lasti+1):i,1:nd)./sc,1, ...
            covchain,meanchain,wsum);
        else
          [covchain,meanchain,wsum] = covupd(par_sim{j}((lasti+1):i,1:nd),1, ...
            covchain,meanchain,wsum);
        end
    
    	upcov = covchain; % update covariance based on past samples
    
      [Ra,p] = chol(upcov);
      if p % singular
        % try to blow it
        [Ra,p] = chol(upcov + eye(nd)*qcov_adjust);
        if p == 0 % choleski decomposition worked
          % scale R
          R = Ra * qcov_scale;
        end
      else
        R = Ra * qcov_scale;
      end
      
      lasti = i;
      
    end
	
end

fintime(j) = cputime;

end%parfor
time_HMC_AM=mean(fintime-initime);

save('/home/pgrad1/1106725p/TransferredFromEuclid5/For_Mihaela_NewProject/Control_21vessels_NonLinear/GPMCMC_NN_sampling_RealData_widerStiff.mat')


for i=1:10
    figure(i); clf(i)
    x = par_sim{i}(101:end,1); postmean(i,1) = mean(x);
    subplot(3,3,1)
    plot(1:5000, x);hold on
    %i=1; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 1');%ylim([l(i) u(i)])
    
    x = par_sim{i}(101:end,2); postmean(i,2) = mean(x);
    subplot(3,3,2)
    plot(1:5000, x); hold on
    %i=2; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 2');%ylim([l(i) u(i)])
    
    x = par_sim{i}(101:end,3); postmean(i,3) = mean(x);
    subplot(3,3,3)
    plot(1:5000, x); hold on
    %i=3; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 3');%ylim([l(i) u(i)])
    
    x = par_sim{i}(101:end,4); postmean(i,4) = mean(x);
    subplot(3,3,4)
    plot(1:5000, x); hold on
    %i=4; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 4');%ylim([l(i) u(i)])
    
    x = par_sim{i}(101:end,5); postmean(i,5) = mean(x);
    subplot(3,3,5)
    plot(1:5000, x); hold on
    %i=3; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 5');%ylim([l(i) u(i)])
    
    x = par_sim{i}(101:end,6); postmean(i,6) = mean(x);
    
    subplot(3,3,6)
    plot(1:5000, x); hold on
    %i=4; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 6');%ylim([l(i) u(i)])
    
    x = par_sim{i}(101:end,7); postmean(i,7) = mean(x);
    
    subplot(3,3,7)
    plot(1:5000, x); hold on
    %i=4; plot(xlim, [par_true(i), par_true(i)], '-r')
    xlabel('Iteration'); ylabel('Param 7');%ylim([l(i) u(i)])
    
end

figure(12);clf(12)
for i=1:10
    title('Log likelihood')
    subplot(3,4,i);plot(1:5100,ObjFct_sim{i})
end

param = [mean(postmean(:,1)),mean(postmean(:,2)), mean(postmean(:,3)), ...
    mean(postmean(:,4)), mean(postmean(:,5))];

% param = [mean(postmean(end,1)),mean(postmean(end,2)), mean(postmean(end,3)), ...
%     mean(postmean(end,4)), mean(postmean(end,5))];

cx = unix(sprintf('./sor06 %f %f %f %f %f %f %f %d', ...
            param(1), param(2), param(3), param(4), param(5), HB, cycles, id));
        
state = CreateData_Optim(id);
pressure = state(end/2+1:end);

figure(11);clf(11)
plot(linspace(0,0.11,512),truePressure, '-k',linspace(0,0.11,512),pressure, '-r')
legend('true pressure', 'generated data')
xlabel('Time');ylabel('data')

figure(1); clf(1)
subplot(3,3,1)
plot(1:4500, par_sim{1}(1:4500,1))
xlabel('Iteration'); ylabel('Param 1')
subplot(3,3,2)
plot(1:4500, par_sim{1}(1:4500,2))
xlabel('Iteration'); ylabel('Param 2')
subplot(3,3,3)
plot(1:4500, par_sim{1}(1:4500,3))
xlabel('Iteration'); ylabel('Param 3')
subplot(3,3,4)
plot(1:4500, par_sim{1}(1:4500,4))
xlabel('Iteration'); ylabel('Param 4')
subplot(3,3,5)
plot(1:4500, par_sim{1}(1:4500,5))
xlabel('Iteration'); ylabel('Param 5')
subplot(3,3,6)
plot(1:4500, par_sim{1}(1:4500,6))
xlabel('Iteration'); ylabel('Param 6')
subplot(3,3,7)
plot(1:4500, par_sim{1}(1:4500,7))
xlabel('Iteration'); ylabel('Param 7')

figure(2); clf(2)
plot(1:4500, ObjFct_sim{1}(1:4500))
xlabel('Iteration'); ylabel('LogLik')

figure(3); clf(3)
subplot(3,3,1); autocorr(par_sim{1}(100:4500,1))
subplot(3,3,2); autocorr(par_sim{1}(100:4500,2))
subplot(3,3,3); autocorr(par_sim{1}(100:4500,3))
subplot(3,3,4); autocorr(par_sim{1}(100:4500,4))
subplot(3,3,5); autocorr(par_sim{1}(100:4500,5))
subplot(3,3,6); autocorr(par_sim{1}(100:4500,6))
subplot(3,3,7); autocorr(par_sim{1}(100:4500,7))

ESS(par_sim{1}(100:4500,:)')

