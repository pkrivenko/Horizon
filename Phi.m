function y = Phi(x)
%global order
order = 2;
n = size(x,1) % sample size
dim = size(x,2) % dim
if order == 1
    y = [ones(n,1), x]
elseif order == 2
    y = zeros(n, 1+dim+dim*(dim+1)/2)
    y(:, 1:2*dim+1) = [ones(n,1), x, x.*x ]
    k = 2*dim+2
    for i = 1:dim-1
        for j = i+1:dim
            y(:, k) = x(:, i).*x(:, j)
            k = k+1
        end
    end
end