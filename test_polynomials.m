clear
clc

dim = 2;
num = 1e4;
deg = 3;
depth = 5;
sample = mvnrnd(zeros(1, dim), eye(dim), num);
y = mvnpdf(sample);
f = @(theta) generatepolynomial2(theta, deg, depth);
for i = 1:num
    x(i,:) = f(sample(i,:));
    if mod(i, 100) == 0
        i
    end
end
b = (x'*x)\(x'*y);
yhat = @(a) f(a)*b;
testf = @(a) -(yhat(a)/mean(y) - mvnpdf(a)/mean(y))^2;
c = fmincon(testf, zeros(1,dim), [],[],[],[],...
    -3*ones(1,dim),3*ones(1,dim));
sims = 1e3;
for i = 1:sims
    a = mvnrnd(zeros(1, dim), eye(dim));
    d(i,1) = yhat(a) - mvnpdf(a);
    if mod(i, 100) == 0
        i
    end
end
ff = @(x, y) mvnpdf([x, y]);
ff2 = @(x,y) yhat([x,y]);
ff3 = @(x,y) ff(x,y)-ff2(x,y);
%ezsurf(ff3, [-3 3], 30)
[X, Y] = meshgrid([-3:0.1:3], [-3:0.1:3]);
for i = 1:size(X,1)
    i
    for j = 1:size(Y,1)
        F3(i,j) = ff3(X(i,j),Y(i,j));
        F1(i,j) = ff(X(i,j),Y(i,j));
        F2(i,j) = ff2(X(i,j),Y(i,j));
    end
end




