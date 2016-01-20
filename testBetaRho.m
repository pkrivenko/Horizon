% generate betaRho for testing reasons
clear all;
global betaRho
betaRho.thetabar = ones(5,1);
betaRho.alpha = .9* eye(5);
betaRho.cov = eye(5);