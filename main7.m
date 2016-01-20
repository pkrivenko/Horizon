function []=main()
clear all; clc; close all;
global betaP betaQ betaV betaS betaB order simulated_grid N_epsd epsd weights_epsd n_theta epsw rho_d si_d Dbar options PHI PHIi beta ga fmintimer s_lb s_ub b_lb b_ub last_sign_s last_sign_b theta_sample n_neighborsPQ Pdata Qdata;
ga = 2; % parameter of CRRA utility
beta = .97^6;
n_iter_v = 5;
n_theta = 300;
N_epsd = 2; % number of quadrature nodes
s_lb = 0;
s_ub = .5;
b_lb = -7;
b_ub = 7;
n_neighbors = 3;
n_neighborsPQ = 3;
last_sign_s = 0; last_sign_b = 0;
%options = optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'off');
options = optimoptions('fmincon', 'Display', 'off');

k_p_up = 1.1;
k_p_down = 1/1.1;
k_b_up = 1.1;
k_b_down = 1/1.1;

simulated_grid = 0; % Use simulated or quadrature grid when calculating means given distributions

n_state_agg = 13; % aggregate state variables (eg moments of distribution)
n_state_ind = 1;  % additional individual state variables (eg individual wealth and age). They always come first: X = (wealth, theta).
order = 2;        % order of polynomials. For now the same for all functions. Later we may want to have higher order for V esp wrt wealth
if order == 1
    n_agg = n_state_agg+1; % number of coefs in P and Q functions
    n_ind = n_state_ind+n_state_agg+1; % number of coefs in V, S, and B functions
elseif order == 2;
    n_agg = 1 + n_state_agg + n_state_agg*(n_state_agg + 1)/2;
    n_ind = 1 + n_state_ind + n_state_agg + (n_state_ind+n_state_agg)*(n_state_ind+n_state_agg + 1)/2;
end

% Conditional distribution of dividends: log(D') = rho_d*log(D) + (1-rho_d)*log(Dbar) + epsd'
Dbar = 1;
rho_d = .5;
si_d = .1;

% Agregate
% State vector theta = [S2 S3 S4 S5 S6 S7 B2 B3 B4 B5 B6 B7 D]
%                        1  2  3  4  5  6  7  8  9 10 11 12 13
% Polynomials Phi  = [1 S2 S3 S4 S5 S6 S7 B2 B3 B4 B5 B6 B7 D ... ]
%                     1  2  3  4  5  6  7  8  9 10 11 12 13 14 ...
% Individual
% State vector w,theta = [w S2 S3 S4 S5 S6 S7 B2 B3 B4 B5 B6 B7 D]
%                         1  2  3  4  5  6  7  8  9 10 11 12 13 14
% Polynomials Phi   =  [1 w S2 S3 S4 S5 S6 S7 B2 B3 B4 B5 B6 B7 D w^2 ... ]
%                       1 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 ...

% Generate S and B sample
Sbar = [ 3   7  15   20   25  30]/100; % S sums to 1, B sums to 0.
Bbar = [-4  -3  -1.5  1.5  3   4];     % Relative scale is also important. Now it's almost arbitrary.

si_s = Sbar/10;
si_b = mean(abs(Bbar))*ones(1,6)/10;

S_sample = randn(n_theta, 6).*(ones(n_theta, 1)*si_s) + ones(n_theta, 1)*Sbar;
B_sample = randn(n_theta, 6).*(ones(n_theta, 1)*si_b) + ones(n_theta, 1)*Bbar;
D_sample = exp(randn(n_theta,1).*(ones(n_theta, 1)*(si_d/(1-rho_d))) + ones(n_theta, 1)*log(Dbar));
theta_sample = [S_sample B_sample D_sample];

% Distance and nearest neighbors
%[neighbors_indexes,distances] = knnsearch(theta_sample,theta_sample,'K',n_neighbors);
neighbors_indexes = find_neighbors(theta_sample);
% neighbors_sample = zeros(n_neighbors, n_state_agg, n_theta);
% for j = 1:n_theta
%     neighbors_sample(:,:,j) = theta_sample(neighbors_indexes(j, 2:n_neighbors),:);
% end

% Generate epsw - eps for current individual wealth (by how much it differs from aggregate wealth for the cohort)
si_w = .1;
seed = 1; rng(seed);
epsw = randn(n_theta,6)*si_w;

% Generate eps_d' - to simulate D' for each D
[N_epsd,epsd,weights_epsd] = GH_Quadrature(N_epsd,1,si_d^2);

% Initial guess on coefficients
Qbar = beta;
premium = 0;
Pbar = Dbar/(1-Qbar+premium);

%betaQ = [log(Qbar) zeros(1, n_agg-1)];
%betaP = [log(Dbar/(1-Qbar+premium)) zeros(1, n_agg-1)];
betaV = [ones(6,2) zeros(6, 13) (-1)*ones(6,1) zeros(6, n_ind - 16)]; % V = 1 + w - w^2

betaS = [zeros(6,2) eye(6) zeros(6, n_ind - 8)];
betaB = [zeros(6,8) eye(6) zeros(6, n_ind - 14)];

Qdata = Qbar * ones(size(theta_sample,1),1);
Pdata = Pbar * ones(size(theta_sample,1),1);
% Compute sample for w
w_sample = ( (Qdata*ones(1,6)).*B_sample + (Pdata*ones(1,6)).*S_sample ) .* exp(epsw);

% Sample of individual state vectors
%x_sample = [w_sample theta_sample];

PHI = Phi(theta_sample);
PHIi = zeros(n_theta, n_ind, 6);
for age = 1:6
    PHIi(:,:,age) = Phi([w_sample(:,age), theta_sample]); % should be different Phi for aggregate and individual state. Or may be embedded into single Phi (check size of state and choose method).
end

Sdata = zeros(n_theta,6); Bdata = Sdata; Vdata = Sdata;

iter = 0;

t_total=tic;
fmintimer = 0;
disp('Iter start')
while (true)
    iter = iter+1
    %pause;
    % Solve individual problem for each point in sample
    for age = 6:(-1):1
        for state = 1:n_theta
            [Sdata(state,age) Bdata(state,age) Vdata(state,age)] = Psi(w_sample(state,age), state, age);
                         disp('    time      iter      age       state'); disp([toc(t_total) iter age state]);
        end
        disp('    time      iter      age      fmintimer'); disp([toc(t_total) iter age fmintimer]);
        betaS(age,:) = (PHIi(:,:,age)\Sdata(:,age))';
        betaB(age,:) = (PHIi(:,:,age)\Bdata(:,age))';
        betaV(age,:) = (PHIi(:,:,age)\Vdata(:,age))';
    end
    
    % Iterate value function
    for iter_v = 1 : n_iter_v
        for age = 6:(-1):1
            for state = 1:n_theta
                Vdata(state,age) = beta * EVprime(Sdata(state,age), w_sample(state,age), theta_sample(state,:), age);
            end
            disp('    time      iter      iter_v    age'); disp([toc(t_total) iter iter_v age fmintimer]);
        end
        %disp('    time      iter      iter_v      fmintimer'); disp([toc(t_total) iter iter_v fmintimer]);
    end
    %disp('    time      iter      fmintimer    k_p_up      k_b_up'); disp([toc(t_total) iter fmintimer     k_p_up      k_b_up]);

    % Update betaQ and betaP
    ZS = sum(Sdata')' - ones(n_theta,1);
    ZB = sum(Bdata')';
    
    [ZS_max, ZS_max_index] = max(ZS); [ZS_min, ZS_min_index] = min(ZS);
    [ZB_max, ZB_max_index] = max(ZB); [ZB_min, ZB_min_index] = min(ZB);
    
    disp('ZS_min     ZS_max     ZB_min    ZB_max'); disp([ZS_min     ZS_max     ZB_min    ZB_max]);

    % Update P
    if sign(ZS_max)==sign(ZS_min)
        if last_sign_s == - sign(ZS_max)
            k_p_up = k_p_up/1.3;
        end
        Pdata = Pdata*k_p_up^(sign(ZS_max));
    else
        ZS_max_neighbors = neighbors_indexes(ZS_max_index,:);
        ZS_min_neighbors = neighbors_indexes(ZS_min_index,:);
        for i = 1:n_neighbors
            index = ZS_max_neighbors(i); Pdata(index) = Pdata(index)*k_p_up;
            index = ZS_min_neighbors(i); Pdata(index) = Pdata(index)/k_p_up;
        end
    end
    
    % Update Q
    if sign(ZB_max)==sign(ZB_min)
        if last_sign_b == - sign(ZB_max)
            k_b_up = k_b_up/1.3;
        end
        Qdata = Qdata*k_b_up^(sign(ZB_max));
    else
        ZB_max_neighbors = neighbors_indexes(ZB_max_index,:);
        ZB_min_neighbors = neighbors_indexes(ZB_min_index,:);
        for i = 1:n_neighbors
            index = ZB_max_neighbors(i);
            Qdata(index) = Qdata(index)*k_b_up;
            index = ZB_min_neighbors(i);
            Qdata(index) = Qdata(index)/k_b_up;
        end
    end
end
end

% Expected value function tomorrow and policy s and b (stocks and bonds held from end of day today to begin of day tomorrow).
% All the arguments (incl age) are for today.
function [s b v] = Psi(w, theta_index, age)
global betaQ betaP betaV betaB betaS epsd rho_d Dbar N_epsd rho_d si_d Dbar weights_epsd options PHI PHIi beta ga fmintimer s_lb s_ub b_lb b_ub theta_sample n_neigborsPQ neighbors Pdata Qdata
negativeEVprime_s = @(s)(-EVprime(s, w, theta_index, age));
t = cputime; % time fminunc separately from everything else
%[s, v] = fminunc(negativeEVprime_s,.1,options);
[s, v] = fmincon(negativeEVprime_s,.1,[],[],[],[],s_lb,s_ub, [], options);
fmintimer = fmintimer + cputime - t;
v = -beta*v;
b = (w - Pdata(theta_index)*s)/(Qdata(theta_index)); % later can use all precomputed PHI and PHIi but need to know the observation in theta_sample
end

% Expected value function tomorrow for given policy s. All the arguments (incl age) are for today.
function y = EVprime(s, w, theta_index, age)
global betaQ betaP betaV betaB betaS epsd rho_d Dbar N_epsd rho_d si_d Dbar weights_epsd PHI PHIi ga theta_sample n_neigborsPQ neighbors Pdata Qdata
theta = theta_sample(theta_index,:);
D = theta(13);

Sprime = S([w theta]); % all others' stock holdings tomorrow
Bprime = B([w theta]); % all others' bond holdings tomorrow
Dprime = exp(rho_d*log(D) + (1-rho_d)*log(Dbar) + epsd);

Vprime = zeros(N_epsd,1);
for k = 1:N_epsd
    thetaprime = [Sprime Bprime Dprime(k)];
    Pprime = Ptheta(Pdata, thetaprime);
    % Qprime = Phi(thetaprime)*betaQ'; % not needed
    wprime = (Pprime+Dprime(k))*s + (w - Pdata(theta_index)*s)/(Qdata(theta_index));
    Vprime(k) = V([wprime thetaprime], age+1);
end

y = weights_epsd'*Vprime;

end

function y = find_neighbors(theta)
global theta_sample n_neighborsPQ
[y,distances] = knnsearch(theta,theta_sample,'K',n_neighborsPQ);
end


function y = Qneigbors(Qdata, neighbors)
y = mean(Qdata(neighbors'));
end

function y = Pneighbors(Pdata, neighbors)
y = mean(Pdata(neighbors'));
end

function y = Qtheta(Qdata, theta)
global theta_sample n_neigborsPQ
neighbors = find_neighbors(theta);
y = Qneighbors(Qdata, neighbors);
end

function y = Ptheta(Pdata, theta)
global theta_sample n_neigborsPQ
%y = transform(Phi(x)*betaP');
neighbors = find_neighbors(theta);
y = Pneighbors(Pdata, neighbors);
end

function y=transform(x)
if x<=0
    y = exp(x);
else
    y = x+1;
end
end

function x=transform_inv(y)
if y<=1
    x = log(y);
else
    x = y-1;
end
end

% function y = S(x,age)
% global betaS
% y = S(x);
% y = y(age);
% end
% 
% function y = B(x,age)
% global betaB
% y = B(x);
% y = y(age);
% end

function y = S(x)
global betaS
y = Phi(x)*betaS';
end

function y = B(x)
global betaB
y = Phi(x)*betaB';
end

function y = V(x,age)
global betaV ga
if age<=6
    y = Phi(x)*betaV(age,:)';
elseif age==7
    y = u(x(1));
else
    throw('Age cannot be greater than 7')
end
end

function y = Phi(x) % Generate polynomials
global order
n = size(x,1); % sample size
dim = size(x,2); % dim
if order == 1
    y = [ones(n,1), x];
elseif order == 2
    y = zeros(n, 1+dim+dim*(dim+1)/2);
    y(:, 1:2*dim+1) = [ones(n,1), x, x.*x ];
    k = 2*dim+2;
    for i = 1 : dim-1
        for j = i+1 : dim
            y(:, k) = x(:, i).*x(:, j);
            k = k+1;
        end
    end
end
end

function y = u(c)
global ga
if ga==1
    y = log(c);
else
    y = (c^(1-ga) - 1) /(1-ga);
end

end