% First run D:\matlab\TE_Leo\Fig_HitvsCR_links_evolution.m to calculate
% links_indices_go, links_indices_nogo

% import channel labels
[FileName,PathName] = uigetfile('*.mat','Select the channel labels file (*.mat)');
load(fullfile(PathName,FileName))
% import network spring layout
[FileName,PathName] = uigetfile('*.mat','Select the network layout file (*.mat)');
load(fullfile(PathName,FileName))

%%

sessions= 1:2;
% compute number of links present in Hit trials
indices_sum = links_indices_go(:,:,1:end) - links_indices_nogo(:,:,1:end);
% --- exclusive links for Hit ---
network_links_hit = squeeze(sum(sum(indices_sum == 1,1),2));

% test for the repeated links in 2 sessions
test_go = (indices_sum == 1);
test_all_go= all(test_go(:,:,sessions),3);

test_nogo = (indices_sum == -1);
test_all_nogo= all(test_nogo(:,:,sessions),3);

figure

subplot(2,1,1)
hold on
axis square

network_plot = plot(digraph(test_all_go,channel_labels));
network_plot.XData = layout_spring_coordinates.X;
network_plot.YData = layout_spring_coordinates.Y;

subplot(2,1,2)
hold on
axis square

network_plot = plot(digraph(test_all_nogo,channel_labels));
network_plot.XData = layout_spring_coordinates.X;
network_plot.YData = layout_spring_coordinates.Y;

hold off

%%
links_mean= mean(unique_links_expert_Hit,1);
err = std(unique_links_expert_Hit,1);

% column is for Hit trial
% column 1:2 learning (column 1 hippocampus column 2 thalamus)
% column 3:4 expert (column 3 hippocampus column 4 thalamus)

groups_Hit = [unique_links_learning_Hit unique_links_expert_Hit];
groups_CR = [unique_links_learning_CR unique_links_expert_CR];
groups= [groups_Hit groups_CR];
[~,~,stats] = anova1(groups);

stats.gnames=['HL Hit'; 'TL Hit'; 'HE Hit'; 'TE Hit'; 'HL CR '; 'TL CR '; 'HE CR '; 'TE CR '];
%%
[c,~,~,gnames] = multcompare(stats);