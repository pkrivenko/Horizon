function []=main()
clear all; clc; close all;
global betaP betaQ betaV betaS betaB order simulated_grid n_epsd epsd weights_epsd n_theta epsw rho_d si_d Dbar options PHI PHIi beta ga fmintimer s_lb s_ub b_lb b_ub last_sign_s last_sign_b theta_sample n_neighborsPQ Pdata Qdata;
ga = 2; % parameter of CRRA utility
beta = .97^6;
n_iter_v = 5;
n_theta = 300;
n_epsd = 5; % number of quadrature nodes
s_lb = 0;
s_ub = .5;
b_lb = -500;
b_ub = 500;
n_neighbors = 5;
n_neighborsPQ = 3;
last_sign_s = 0; last_sign_b = 0;
%options = optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'off');
options = optimoptions('fmincon', 'Display', 'off');

k_p_up = 2;
k_b_up = 2;

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
Bbar = [-3  -2  -1  1  2   3]/20;     % Relative scale is also important. Now it's almost arbitrary.

si_s = Sbar/10;
si_b = mean(abs(Bbar))*ones(1,6)/10;

S_sample = randn(n_theta, 6).*(ones(n_theta, 1)*si_s) + ones(n_theta, 1)*Sbar;
B_sample = randn(n_theta, 6).*(ones(n_theta, 1)*si_b) + ones(n_theta, 1)*Bbar;
D_sample = exp(randn(n_theta,1).*(ones(n_theta, 1)*(si_d/(1-rho_d))) + ones(n_theta, 1)*log(Dbar));
theta_sample = [S_sample B_sample D_sample];

for state = 1:n_theta
    theta_sample(state,:) = fe(theta_sample(state,:));
end

% Distance and nearest neighbors
%[neighbors_indexes,distances] = knnsearch(theta_sample,theta_sample,'K',n_neighbors);
neighbors_indexes = find_neighbors(theta_sample, n_neighbors);
% neighbors_sample = zeros(n_neighbors, n_state_agg, n_theta);
% for j = 1:n_theta
%     neighbors_sample(:,:,j) = theta_sample(neighbors_indexes(j, 2:n_neighbors),:);
% end

% Generate epsw - eps for current individual wealth (by how much it differs from aggregate wealth for the cohort)
si_w = .1;
seed = 1; rng(seed);
epsw = randn(n_theta,6)*si_w;

% Generate eps_d' - to simulate D' for each D
[n_epsd,epsd,weights_epsd] = GH_Quadrature(n_epsd,1,si_d^2);

% Initial guess on coefficients
Qbar = beta;
premium = 0;
Pbar = Dbar/(1-Qbar+premium);

%betaQ = [log(Qbar) zeros(1, n_agg-1)];
%betaP = [log(Dbar/(1-Qbar+premium)) zeros(1, n_agg-1)];
betaV = [ones(6,2) zeros(6, 13) (-1/15)*ones(6,1) zeros(6, n_ind - 16)]; % V = 1 + w - w^2

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
    iter = iter+1;
    % Solve individual problem for each point in sample
    for state = 1:n_theta
        [Sdata(state,:), Bdata(state,:), Vdata(state,:)] = Psi(w_sample(state,:), state);
        %disp('    time      iter      age       state'); disp([toc(t_total) iter age state]);
    end
    
    for age = 1:6
        betaS(age,:) = (PHIi(:,:,age)\Sdata(:,age))';
        betaB(age,:) = (PHIi(:,:,age)\Bdata(:,age))';
        betaV(age,:) = (PHIi(:,:,age)\Vdata(:,age))';
    end
    
%     % Iterate value function
%     for iter_v = 1 : n_iter_v
%         for age = 6:(-1):1
%             for state = 1:n_theta
%                 Vdata(state,age) = beta * EVprime(Sdata(state,age), w_sample(state,age), theta_sample(state,:), age);
%             end
%             disp('    time      iter      iter_v    age'); disp([toc(t_total) iter iter_v age fmintimer]);
%         end
%         %disp('    time      iter      iter_v      fmintimer'); disp([toc(t_total) iter iter_v fmintimer]);
%     end
%     %disp('    time      iter      fmintimer    k_p_up      k_b_up'); disp([toc(t_total) iter fmintimer     k_p_up      k_b_up]);

    % Update betaQ and betaP
    ZS = sum(Sdata,2) - ones(n_theta,1);
    ZB = sum(Bdata,2);
    
    [ZS_max, ZS_max_index] = max(ZS); [ZS_min, ZS_min_index] = min(ZS);
    [ZB_max, ZB_max_index] = max(ZB); [ZB_min, ZB_min_index] = min(ZB);
    
    % disp('    time      iter      fmintimer'); 
    disp([toc(t_total) iter fmintimer]);
    %disp('ZS_min     ZS_max     ZB_min    ZB_max    time      iter      fmintimer      ');
    %disp([ZS_min              ZS_max              ZB_min              ZB_max       toc(t_total) iter fmintimer]);
    %disp([ZS_min_index        ZS_max_index        ZB_min_index        ZB_max_index; ...
    %Pdata(ZS_min_index) Pdata(ZS_max_index) Pdata(ZB_min_index) Pdata(ZB_max_index);...
    %Qdata(ZS_min_index) Qdata(ZS_max_index) Qdata(ZB_min_index) Qdata(ZB_max_index)]);
    
    indexes_old = [ZS_min_index        ZS_max_index        ZB_min_index        ZB_max_index];
    print_data = [ZS_min              ZS_max              ZB_min              ZB_max       ;...
                  ZS_min_index        ZS_max_index        ZB_min_index        ZB_max_index; ...
            Pdata(ZS_min_index) Pdata(ZS_max_index) Pdata(ZB_min_index) Pdata(ZB_max_index);...
            Qdata(ZS_min_index) Qdata(ZS_max_index) Qdata(ZB_min_index) Qdata(ZB_max_index)];
    
    rowheadings={'Z', 'Index', 'P', 'Q'}; wid = 8; fms='.4f'; fileID=1; colsep=' ';
    colheadings={'ZS_min', 'ZS_max', 'ZB_min', 'ZB_max'};
    displaytable(print_data,colheadings,wid,fms,rowheadings,fileID,colsep);
    
    
    
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
function [s, b, v] = Psi(w, theta_index)
global betaQ betaP betaV betaB betaS epsd rho_d Dbar n_epsd rho_d si_d Dbar weights_epsd options PHI PHIi beta ga fmintimer s_lb s_ub b_lb b_ub theta_sample n_neighborsPQ neighbors Pdata Qdata

theta = theta_sample(theta_index,:);
D = theta(13);
Dprime = exp(rho_d*log(D) + (1-rho_d)*log(Dbar) + epsd);
[Sprime, Bprime] = SBaggregate(theta_index); % Aggregate S and B TOMORROW decided TODAY
Sprime = repmat(Sprime, n_epsd, 1); % aggregate s
Bprime = repmat(Bprime, n_epsd, 1); % aggregate b
theta_prime = [Sprime Bprime Dprime];
neighbors_prime = find_neighbors(theta_prime, n_neighborsPQ);
v = zeros(1,6); s = v; b = v;
for age = 1:6
    negativeEVprime_s = @(s)(-EVprime(s, w(age), theta_index, age, theta_prime, neighbors_prime));
    t = cputime; % time fminunc separately from everything else
    [s(age), v(age)] = fmincon(negativeEVprime_s,.1,[],[],[],[],s_lb,s_ub, [], options);
    fmintimer = fmintimer + cputime - t;
    v(age) = -beta*v(age);
    b(age) = (w(age) - Pdata(theta_index)*s(age))/(Qdata(theta_index)); % later can use all precomputed PHI and PHIi but need to know the observation in theta_sample
    b(age) = max(min(b(age), b_ub), b_lb);
end

end

% Expected value function tomorrow for given policy s. All the arguments (incl age) are for today.
function y = EVprime(s, w, theta_index, age, theta_prime, neighbors_prime)
global betaQ betaP betaV betaB betaS epsd rho_d Dbar n_epsd rho_d si_d Dbar weights_epsd PHI PHIi ga theta_sample n_neigborsPQ Pdata Qdata
theta = theta_sample(theta_index,:);
D = theta(13);

Sprime = theta_prime(:, 1:6);  % all others' stock holdings tomorrow
Bprime = theta_prime(:, 7:12); % all others' bond holdings tomorrow
Dprime = theta_prime(:, 13);

Vprime = zeros(n_epsd,1);
for k = 1:n_epsd
    thetaprime = theta_prime(k,:);
    Pprime = Pneighbors(Pdata, neighbors_prime(k,:));
    % Qprime = Phi(thetaprime)*betaQ'; % not needed
    wprime = (Pprime+Dprime(k))*s + (w - Pdata(theta_index)*s)/(Qdata(theta_index));
    Vprime(k) = V([wprime thetaprime], age+1);
end

y = weights_epsd'*Vprime;

end

function y = find_neighbors(theta, n)
global theta_sample
[y,distances] = knnsearch(theta,theta_sample,'K',n);
end


function y = Qneigbors(Qdata, neighbors)
y = mean(Qdata(neighbors'));
end

function y = Pneighbors(Pdata, neighbors)
y = mean(Pdata(neighbors'));
end

function y = Qtheta(Qdata, theta)
global theta_sample n_neigborsPQ
neighbors = find_neighbors(theta, n_neigborsPQ);
y = Qneighbors(Qdata, neighbors);
end

function y = Ptheta(Pdata, theta)
global theta_sample n_neigborsPQ
%y = transform(Phi(x)*betaP');
neighbors = find_neighbors(theta, n_neigborsPQ);
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

function [Sprime Bprime] = SBaggregate(theta_index)
global betaS betaB theta_sample Pdata
theta = theta_sample(theta_index,:);
S = theta(:, 1:6);
B = theta(:, 7:12);
D = theta(:, 13);
P = Pdata(theta_index);
% Q = Qdata(theta_index);
w = (P+D).*S+B;
Sprime = zeros(1,6); Bprime = Sprime; BprimePlus=0; BprimeMinus=0;
for age = 1:6
    Sprime(age) = Phi([w(age) theta])*betaS(age,:)';
    Bprime(age) = Phi([w(age) theta])*betaB(age,:)';
    if Bprime(age)>0
        BprimePlus = BprimePlus + Bprime(age);
    else
        BprimeMinus = BprimeMinus - Bprime(age);
    end
end
% Force equilibrium
Sprime = Sprime / sum(Sprime);
if BprimePlus > BprimeMinus % excess saving
   for age = 1:6
       if Bprime(age)>0
           Bprime(age)=Bprime(age)*BprimeMinus/BprimePlus;
       end
   end
elseif BprimeMinus>0
    for age = 1:6
        if Bprime(age)<0
            Bprime(age)=Bprime(age)*BprimePlus/BprimeMinus;
        end
    end
end
end

function y = V(x,age)
global betaV ga
if age<=6
    p = Phi(x);
    q = betaV(age,:);
    y = p*q';
    b = q(2) + p(3:15)*q(30:42)';
    b = -b/2/q(16);
    if p(2) > b
        p = Phi([b, x(2:end)]);
        y = p*q';
    end
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

% y = 10*c - c^2;

if ga==1
    y = log(c);
else
    y = (c.^(1-ga) - 1) /(1-ga);
end

%y = -.723 + .78*c - .125*c.^2 + .0064*c.^3;

end

% Force equilibrium
function theta_fe = fe(theta)
S = theta(:, 1:6);
B = theta(:, 7:12);
S = S / sum(S);
BprimePlus=0; BprimeMinus=0;
for age = 1:6
    if B(age)>0
        BprimePlus = BprimePlus + B(age);
    else
        BprimeMinus = BprimeMinus - B(age);
    end
end
if BprimePlus > BprimeMinus % excess saving
   for age = 1:6
       if B(age)>0
           B(age)=B(age)*BprimeMinus/BprimePlus;
       end
   end
elseif BprimeMinus>0
    for age = 1:6
        if B(age)<0
            B(age)=B(age)*BprimePlus/BprimeMinus;
        end
    end
end
theta_fe = [S B theta(:,13)];
end