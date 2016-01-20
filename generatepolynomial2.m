function f = generatepolynomial2(theta, degree, depth)
n = length(theta);
d = bsxfun(@minus, nchoosek(0:n+degree-1,degree), 0:degree-1)+1;
t = [1, theta];
for i = 1:length(d)
    f(i) =  1;
    for p = d(i, :)
        f(i) = f(i)*t(p);
    end
end
for k = 1:depth
    for j = 1:n
        i = i+1;
        f(i) = theta(j)^(degree+k);
    end
end