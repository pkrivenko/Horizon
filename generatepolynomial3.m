function f = generatepolynomial3(dim, degree, depth)
% The function generate the 1-by-? cell array of 
% function handle polynomials for dim-dimensional 
% vector of inputs with all interaction terms up to
% degree specified by degree and additional powers of
% single variables up to degree degree+depth.
% Examples:
% f = generatepolynomial3(5, 3, 2);
% f = generatepolynomial3(6, 3, 0);
% f = generatepolynomial3(3, 0, 2);
% for i = 1:length(f)
%   a(i) = f{i}([1,2,3]);
% end
%
% f = generatepolynomial3(3, 0, 2);
% x = [0.5, 0.2, 1.3];
% g = @(myfunction) myfunction(x);
% cellfun(g, f)
% ans =
%
%    0.5000    0.2000    1.3000    0.2500    0.0400    1.6900


% f = generatepolynomial3(10, 2, 0);
% x = [0.5, 0.2, 1.3, 10, -37, 0.5, 0.2, 1.3, 10, -37];
% g = @(myfunction) myfunction(x);
% cellfun(g, f)

n = dim;
d = bsxfun(@minus, nchoosek(0:n+degree-1,degree), 0:degree-1);
t = @(theta, p) (p==0) + (p~=0)*theta(p+(p==0));
f = cell(1, length(d)+depth*n);
for i = 1:length(d)
%    [i length(d)]
    g = @(theta) 1;
    for p = d(i, :)
        g = @(theta) g(theta)*t(theta,p);
    end
    f{i} = g;
end
for k = 1:depth
    for j = 1:n
        i = i+1;
%        i
        if isempty(i)
            i = 1;
        end
        g = @(theta) theta(j)^(degree+k);
        f{i} = g;
    end
end