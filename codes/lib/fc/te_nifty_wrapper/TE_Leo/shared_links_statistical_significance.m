%% Test analytical solution against Monte Carlo simulation

links_total = 132;
links1 = 20;
links2 = 52;

% Analytical solution

possible_links_shared = max(0,links1+links2-links_total):min(links1,links2);

p_values = NaN([1,length(possible_links_shared)]);
count = 1;
for links_shared = possible_links_shared
    p_values(count) = nchoosek(links_total,links_shared)*nchoosek(links_total-links_shared,links1-links_shared)*nchoosek(links_total-links1,links2-links_shared)/(nchoosek(links_total,links1)*nchoosek(links_total,links2));
    count = count + 1;
end

% Monte Carlo simulation

MC_trials = 500000;

network1 = [ones([1,links1]),zeros([1,links_total - links1])];
network2 = [ones([1,links2]),zeros([1,links_total - links2])];

results_count = NaN([1,MC_trials]);
for count_trials = 1:MC_trials

    network1 = network1(randperm(links_total));
    network2 = network2(randperm(links_total));
    
    results_count(count_trials) = sum(sum(network1 + network2 == 2));
    
end


%% plot

figure()

subplot(1,2,1)
bar(possible_links_shared,p_values)
xlabel('Number of shared links')
ylabel('p')
title('Analytical solution')

subplot(1,2,2)
histogram(results_count,possible_links_shared,'Normalization','probability')
xlabel('Number of shared links')
ylabel('p')
title('Monte Carlo')


