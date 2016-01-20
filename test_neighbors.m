% test [neighbors_sample,distances] = knnsearch(theta_sample,theta_sample,'K',n_neighbors);

n_theta = 100;
n_neighbors = 30;
theta = 36;

theta_sample = randn(n_theta,2);

[neighbors_indexes,distances] = knnsearch(theta_sample,theta_sample,'K',n_neighbors);

neighbors_sample = theta_sample(neighbors_indexes(theta, 2:n_neighbors),:);

scatter(theta_sample(:,1), theta_sample(:,2)); hold on;
scatter(theta_sample(theta,1), theta_sample(theta,2), 'red');
scatter(neighbors_sample(:,1), neighbors_sample(:,2), 'green');

hold off;