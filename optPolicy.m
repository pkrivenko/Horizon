% Policy function for an individual agent.
% 
% Inputs
% w - 1x1 wealth
% tau - 1x1 'age' = time left before retirement
% theta - ?x1 aggregate state.
% 
% Output
% s - 1x1 stock tomorrow, i.e. S(t+1)
% 
% Other functions used
% P(theta), Q(theta), ValueFunction(w,tau,theta)
% 
% Global variables used directly (not only in other functions above)
% betaRho should consist of betaRho.thetabar (?x1), betaRho.alpha (?x?) and betaRho.cov (symmetric ?x? cov matrix)
% 

function s = optPolicy(w, tau, theta)
global betaV betaRho betaP betaQ

% (negative of) EVprime as a function of S(t+1)
negativeEVprime = @(s)(-EVprime(w,tau,theta,s));

s = fminunc(negativeEVprime,0);

end

% Expected value function TOMORROW given wealth, 'age', and state TODAY, AND stock chosen for tomorrow
function y = EVprime(w,tau,theta,s)
global betaV betaRho betaP betaQ
[ThetaPrime, weight_nodes, Nnodes] = thetaPrimeGrid(theta);
VPrime = zeros(Nnodes,1);
parfor node = 1:Nnodes
    thetaPrime=ThetaPrime(node,:)';
    wPrime = (P(thetaPrime)+thetaPrime(end)) * s + (w - P(theta)*s) / Q(theta);
    VPrime(node) = ValueFunction(wPrime,tau-1,thetaPrime);
end
y = VPrime'*weight_nodes;
end

function [ThetaPrime, weight_nodes, Nnodes] = thetaPrimeGrid(theta)
global betaRho
Qn = 3; % # of gridpoints for each dimension of theta. Total # gridpoints is Qn^(dim of theta). For Qn=4 and 5-dim theta it's 5^4 = 625.
sizeTheta = size(betaRho.cov,1);
[Nnodes,epsi_nodes,weight_nodes] = GH_Quadrature(Qn,sizeTheta,betaRho.cov); % this can be precalculated - don't have to redo for every theta
ThetaPrime = kron( (betaRho.thetabar + betaRho.alpha * (theta - betaRho.thetabar) )' , ones(Nnodes,1)) + epsi_nodes;
end