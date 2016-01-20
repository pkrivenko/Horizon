function []=main()
global betaP betaQ betaV betaS betaB order simulated_grid N_epsd epsd weights_epsd N_theta epsw rho_d si_d Dbar options;
t_total=tic;
ga = 2; % parameter of CRRA utility

simulated_grid = 1; % Use simulated or quadrature grid when calculating means given distributions

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
rho_d = .9;
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
N_theta = 300;
Sbar = [ 3   7  15   20   25  30]/100; % S sums to 1, B sums to 0.
Bbar = [-4  -3  -1.5  1.5  3   4];     % Relative scale is also important. Now it's almost arbitrary.

si_s = Sbar/10;
si_b = mean(abs(Bbar))*ones(1,6)/10;

S_sample = randn(N_theta, 6).*(ones(N_theta, 1)*si_s) + ones(N_theta, 1)*Sbar;
B_sample = randn(N_theta, 6).*(ones(N_theta, 1)*si_b) + ones(N_theta, 1)*Bbar;
D_sample = exp(randn(N_theta,1).*(ones(N_theta, 1)*(si_d/(1-rho_d))) + ones(N_theta, 1)*log(Dbar));
theta_sample = [S_sample B_sample D_sample];

% Generate epsw - eps for current individual wealth (by how much it differs from aggregate wealth for the cohort)
si_w = .1;
seed = 1; rng(seed);
epsw = randn(N_theta,6)*si_w;

% Generate eps_d' - to simulate D' for each D
N_epsd = 5; % number of quadrature nodes
[N_epsd,epsd,weights_epsd] = GH_Quadrature(N_epsd,1,si_d^2);

% Initial guess on coefficients
Qbar = .97^6;
premium = 0;
betaQ = [Qbar zeros(1, n_agg-1)];
betaP = [Dbar/(1-Qbar+premium) zeros(1, n_agg-1)];
betaV = [ones(6,2) zeros(6, 13) (-1)*ones(6,1) zeros(6, n_ind - 16)]; % V = 1 + w - w^2

betaS = [zeros(6,2) eye(6) zeros(6, n_ind - 8)];
betaB = [zeros(6,8) eye(6) zeros(6, n_ind - 14)];

% Compute sample for w
w_sample = ( (Q(theta_sample)*ones(1,6)).*B_sample + (P(theta_sample)*ones(1,6)).*S_sample ) .* exp(epsw);

% Sample of individual state vectors
%x_sample = [w_sample theta_sample];

Sdata = zeros(N_theta,6); Bdata = Sdata; Vdata = Sdata;
PHIi = zeros(N_theta, n_ind, 6);

options = optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'off');
iter = 0;
while (true)
    iter = iter+1;
    % Solve individual problem for each point in sample
    for k = 1:6
        for j = 1:N_theta
            [Sdata(j,k) Bdata(j,k) Vdata(j,k)] = Psi(w_sample(j,k), theta_sample(j,:));
                disp('    time      iter      k         j');
                disp([toc(t_total) iter k j]);
        end
        PHIi(:,:,k) = Phi([w_sample(:,k), theta_sample]); % should be different Phi for aggregate and individual state. Or may be embedded into single Phi (check size of state and choose method).
    end
    
    %PHI = Phi(theta_sample);
    
    
    % Fit betas
%     betaS = repmat(PHIi\Sdata', 1, 6)';
%     betaB = repmat(PHIi\Bdata', 1, 6)';
%     betaV = repmat(PHIi\Vdata', 1, 6)';
    betaS = PHIi\Sdata';
    betaB = PHIi\Bdata';
    betaV = PHIi\Vdata';
    
    %Update betaQ and BetaP
    %    ...
    %    betaQ = PHI\Q';
    %    betaP = PHI\P';
end

end

function [s b v] = Psi(w, theta)
global betaQ betaP betaV betaB betaS epsd rho_d Dbar N_epsd rho_d si_d Dbar weights_epsd options
negativeEVprime_s = @(s)(negativeEVprime(s, w, theta));
[s, v] = fminunc(negativeEVprime_s,.1,options);
v = -v;
b = (w - Phi(theta)*betaP'*s)/(Phi(theta)*betaQ');
end

function y = negativeEVprime(s, w, theta)
global betaQ betaP betaV betaB betaS epsd rho_d Dbar N_epsd rho_d si_d Dbar weights_epsd

%S = theta(1:6);
%B = theta(7:12);
D = theta(13);

Sprime = Phi([w theta])*betaS';
Bprime = Phi([w theta])*betaB';
Dprime = exp(rho_d*log(D) + (1-rho_d)*log(Dbar) + epsd);

Vprime = zeros(N_epsd,1);
for k = 1:N_epsd
    thetaprime = [Sprime Bprime Dprime(k)];
    Pprime = Phi(thetaprime)*betaP';
    % Qprime = Phi(thetaprime)*betaQ';
    wprime = (Pprime+Dprime(k))*s + (w - Phi(theta)*betaP'*s)/(Phi(theta)*betaQ');
    Vprime(k) = - Phi([wprime thetaprime]) * betaV(3,:)';
end

y = weights_epsd'*Vprime;

end

function y = Q(x)
global betaQ
y = Phi(x)*betaQ';
end

function y = P(x)
global betaP
y = Phi(x)*betaP';
end

function y = V(x)
global betaV
y = Phi(x)*betaV';
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