clear all; clc;
f = generatepolynomial3(5, 2, 0);
syms x1 x2 x3 x4 x5
for i = 1:length(f)
    a(i) = f{i}([x1 x2 x3 x4 x5]);
end
a