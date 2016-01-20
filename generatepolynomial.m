function f = generatepolynomial(theta, degree)
n = length(theta);
d = bsxfun(@minus, nchoosek(0:n+degree-1,degree), 0:degree-1)+1;
t = [1, theta];
for i = 1:length(d)
    f(i) =  1;
    for p = d(i, :)
        f(i) = f(i)*t(p);
    end
end