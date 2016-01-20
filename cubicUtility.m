% cubic utility
function betaU = cubicUtility()
global ga
ga = 2;
cMin = .5;
cMax = 10;
n = 1000;

step = (cMax-cMin)/(n-1);

c = (cMin:step:cMax)';

y = u(c);

X = [ones(n,1) c c.^2 c.^3];

betaU = X\y;

yhat = X*betaU;

plot(c,y,c,yhat);

end

function y = u(c)
global ga

if ga==1
    y = log(c);
else
    y = (c.^(1-ga) - 1) /(1-ga);
end

end