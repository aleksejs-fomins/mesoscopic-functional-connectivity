%% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m

addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\mtp\');
addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\TE_Leo\');

%% plot TE and p-value for a specific channel pair

% choose channel pair to plot
channel1 = 1;
channel2 = 8;

% check if the desired channel pair was evalued
if isempty(results.TE_table{channel1,channel2}{1,1})
    error('error: no results found for the desired channel pair')
end

plot_x_limits = results.parameters.samples_original([1,end]) / results.parameters.sample_rate;
bar_spacing = 0.01;

% plot trial-averaged calcium traces and TE
figure
subplot(3,1,1)
% create channel1 bar plot
plot_x = results.parameters.samples_timesteps;
plot_y = mean(results.data(:,:,channel1),1);
hold on
for count = 1:length(plot_x)
    bar_x = plot_x(count);
    bar_y = plot_y(count);
    if results.parameters.subsampling
        bar_width = results.parameters.subsampling_windows_lenghts(count) / results.parameters.sample_rate;
    else
        bar_width = 1 / results.parameters.sample_rate;
    end
    h = bar(bar_x + bar_width/2,bar_y,bar_width - bar_spacing);    
end
% overlay plot of bin edges
plot(plot_x,results.states_bins_edges{1,channel1}{1,1})
xlim(plot_x_limits)
ylim([min(min(results.states_bins_edges{1,channel1}{1,1}(:,2:end-1))),max(max(results.states_bins_edges{1,channel1}{1,1}(:,2:end-1)))])
ylabel(results.parameters.channel_labels(channel1))
title('Calcium level (average over trials)')
hold off

subplot(3,1,2)
% create channel2 bar plot
plot_x = results.parameters.samples_timesteps;
plot_y = mean(results.data(:,:,channel2),1);
hold on
for count = 1:length(plot_x)    
    bar_x = plot_x(count);
    bar_y = plot_y(count);
    if results.parameters.subsampling
        bar_width = results.parameters.subsampling_windows_lenghts(count) / results.parameters.sample_rate;
    else
        bar_width = 1 / results.parameters.sample_rate;
    end        
    h = bar(bar_x + bar_width/2,bar_y,bar_width - bar_spacing);
end
% overlay plot of bin edges
plot(plot_x,results.states_bins_edges{1,channel2}{1,1})
xlim(plot_x_limits)
ylim([min(min(results.states_bins_edges{1,channel2}{1,1}(:,2:end-1))),max(max(results.states_bins_edges{1,channel2}{1,1}(:,2:end-1)))])
ylabel(results.parameters.channel_labels(channel2))
hold off

subplot(3,1,3)
% create TE bar plot with color coding for p-values
plot_x = results.parameters.samples_timesteps;
plot_y = results.TE_table{channel1,channel2}{1,1};
hold on
for count = 1:length(plot_x)    
    bar_x = plot_x(count);
    bar_y = plot_y(count);
    bar_p = results.p_table{channel1,channel2}{1,1}(count);
    if results.parameters.subsampling
        bar_width = results.parameters.subsampling_windows_lenghts(count) / results.parameters.sample_rate;
    else
        bar_width = 1 / results.parameters.sample_rate;
    end
    
    h = bar(bar_x + bar_width/2,bar_y,bar_width-bar_spacing);
    
    % define bar color depending on p-value
    if bar_p < 0.01
        set(h,'FaceColor','g');
    elseif bar_p < 0.05
        set(h,'FaceColor','y');
    else
        set(h,'FaceColor','r');
    end
end
hold off
% bar(results.parameters.samples_timesteps,results.TE_table{channel1,channel2}{1,1})
xlim(plot_x_limits)
ylim([0,max(plot_y)*1.1])
xlabel('Time')
ylabel('TE(bits)')
title(sprintf('%s --> %s (estimated delay = %d)',results.parameters.channel_labels{1,channel1},results.parameters.channel_labels{1,channel2},results.delay_table{channel1,channel2}{1,1}))


%% plot TE table, p-value table, and delay table (all channel pairs)

% plot TE table
figure
for count = 1:length(results.parameters.channel_pairs)
    subplot(length(results.parameters.channel_labels),length(results.parameters.channel_labels),(results.parameters.channel_pairs(count,1)-1)*length(results.parameters.channel_labels)+results.parameters.channel_pairs(count,2))
    plot(results.parameters.samples_timesteps,results.TE_table{results.parameters.channel_pairs(count,1),results.parameters.channel_pairs(count,2)}{1,1})
end

% plot p-value table
figure
for count = 1:length(results.parameters.channel_pairs)    
    subplot(length(results.parameters.channel_labels),length(results.parameters.channel_labels),(results.parameters.channel_pairs(count,1)-1)*length(results.parameters.channel_labels)+results.parameters.channel_pairs(count,2))
    plot(results.parameters.samples_timesteps,results.p_table{results.parameters.channel_pairs(count,1),results.parameters.channel_pairs(count,2)}{1,1})
end

% plot delay table
figure
delay_table = uitable('Data',results.delay_table{:,:},'ColumnName',results.delay_table.Properties.VariableNames,'RowName',results.delay_table.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);


%% evolution of shared links between consecutive training sessions

% define frequency threshold
th = 0;
% find number of nodes and links
network_size = size(network_adj_matrices,1);
network_links_total = network_size^2 - network_size;
% find indices of weights above threshold
links_indices = network_adj_matrices > th;
% compute number of links
network_links = squeeze(sum(sum(links_indices,1),2));

% compute number of shared links between consecutive sessions
indices_sum = links_indices(:,:,1:end-1) + links_indices(:,:,2:end);
network_links_shared = squeeze(sum(sum(indices_sum == 2,1),2));
% compute p-value of shared links
network_links_shared_p_value = NaN([length(network_links_shared),1]);
warning('off','MATLAB:nchoosek:LargeCoefficient');
for count = 1:sessions-1
    network_links1 = network_links(count);
    network_links2 = network_links(count+1);
    network_links_shared_p_value(count) = nchoosek(network_links_total,network_links_shared(count))*nchoosek(network_links_total-network_links_shared(count),network_links1-network_links_shared(count))*nchoosek(network_links_total-network_links1,network_links2-network_links_shared(count))/(nchoosek(network_links_total,network_links1)*nchoosek(network_links_total,network_links2));
end

% plot shared links and total links numbers across sessions
figure
plot_links = plot(2:sessions,network_links(2:end),':bo',2:sessions,network_links_shared,':r');
plot_links(1).MarkerFaceColor = 'b';
plot_links(1).MarkerEdgeColor = 'k';
xlabel('Training sessions')
ylabel('Links')
ylim([0,network_links_total])
legend('Total','Shared with previous session')
% overlay markers with color coding for p-values
marker_color = NaN([length(network_links_shared_p_value),1]);
marker_color(network_links_shared_p_value >= 0.05) = 3;
marker_color(network_links_shared_p_value < 0.05) = 2;
marker_color(network_links_shared_p_value < 0.01) = 1;
hold on
scatter_links = scatter(2:sessions,network_links_shared,40,marker_color,'filled');
scatter_links.MarkerEdgeColor = 'k';
colormap([0,1,0;1,1,0;1,0,0])


% compute percentage of links shared with the previous session (relative to the number of links in
% the current session)
network_links_shared_percentage = network_links_shared./network_links(2:end);

% plot percentage
figure
plot_links = plot(2:sessions,network_links_shared_percentage,':r');
plot_links(1).MarkerEdgeColor = 'k';
xlabel('Training sessions')
ylabel('Shared links with previous session (%)')
%ylim([0,network_links_total])
% overlay markers with color coding for p-values
marker_color = NaN([length(network_links_shared_p_value),1]);
marker_color(network_links_shared_p_value >= 0.05) = 3;
marker_color(network_links_shared_p_value < 0.05) = 2;
marker_color(network_links_shared_p_value < 0.01) = 1;
hold on
scatter_links = scatter(2:sessions,network_links_shared_percentage,40,marker_color,'filled');
scatter_links.MarkerEdgeColor = 'k';
colormap([0,1,0;1,1,0;1,0,0])


%% stability of links

% define frequency threshold
th = 0;
% find number of nodes and links
network_size = size(network_adj_matrices,1);
network_links_total = network_size^2 - network_size;
% find indices of weights above threshold
links_indices = network_adj_matrices > th;
% compute number of links
network_links = squeeze(sum(sum(links_indices,1),2));

% compute number of shared links between each session and the last one
indices_sum = links_indices(:,:,1:sessions) + repmat(links_indices(:,:,sessions),1,1,sessions);
network_links_shared = squeeze(sum(sum(indices_sum == 2,1),2));
% compute p-value of shared links
network_links_shared_p_value = NaN([length(network_links_shared),1]);
warning('off','MATLAB:nchoosek:LargeCoefficient');
for count = 1:sessions
    network_links1 = network_links(count);
    network_links2 = network_links(sessions);
    network_links_shared_p_value(count) = nchoosek(network_links_total,network_links_shared(count))*nchoosek(network_links_total-network_links_shared(count),network_links1-network_links_shared(count))*nchoosek(network_links_total-network_links1,network_links2-network_links_shared(count))/(nchoosek(network_links_total,network_links1)*nchoosek(network_links_total,network_links2));
end

% plot shared links and total links numbers across sessions
figure
plot_links = plot(1:sessions,network_links(1:end),':bo',1:sessions,network_links_shared,':r');
plot_links(1).MarkerFaceColor = 'b';
plot_links(1).MarkerEdgeColor = 'k';
xlabel('Training sessions')
ylabel('Links')
ylim([0,network_links_total])
legend('Total','Shared with last session')
% overlay markers with color coding for p-values
marker_color = NaN([length(network_links_shared_p_value),1]);
marker_color(network_links_shared_p_value >= 0.05) = 3;
marker_color(network_links_shared_p_value < 0.05) = 2;
marker_color(network_links_shared_p_value < 0.01) = 1;
hold on
scatter_links = scatter(1:sessions,network_links_shared,40,marker_color,'filled');
scatter_links.MarkerEdgeColor = 'k';
colormap([0,1,0;1,1,0;1,0,0])

% compute percentage of links shared with the previous session (relative to the number of links in
% the current session)
network_links_shared_percentage = network_links_shared./network_links;

% plot percentage
figure
plot_links = plot(1:sessions,network_links_shared_percentage,':r');
plot_links(1).MarkerEdgeColor = 'k';
xlabel('Training sessions')
ylabel('Shared links with last session (%)')
%ylim([0,network_links_total])
% overlay markers with color coding for p-values
marker_color = NaN([length(network_links_shared_p_value),1]);
marker_color(network_links_shared_p_value >= 0.05) = 3;
marker_color(network_links_shared_p_value < 0.05) = 2;
marker_color(network_links_shared_p_value < 0.01) = 1;
hold on
scatter_links = scatter(1:sessions,network_links_shared_percentage,40,marker_color,'filled');
scatter_links.MarkerEdgeColor = 'k';
colormap([0,1,0;1,1,0;1,0,0])


%% generate and save network layout from experimental adjacency matrix

% load channel labels and weighted adjacency matrix from files
channel_labels = importdata('E:\Downloads\areas_name.xlsx');
adjacency_matrix = importdata('E:\Downloads\projections_12.xlsx');

% choose path to save layout coordinates and imported data as matlab files
layout_save_path = 'E:\Downloads\';

% replace forbidden characters in channel labels
channel_labels = strrep(channel_labels, ' ', '_');
channel_labels = strrep(channel_labels, '/', '_');

% replace NaN entries with zero weight in the adjacency matrix
adjacency_matrix(isnan(adjacency_matrix)) = 0;

% create table from matrix and labels
adjacency_table = array2table(adjacency_matrix, 'VariableNames', channel_labels, 'RowNames', channel_labels );

% create directed graph object
layout_graph = digraph(adjacency_matrix,channel_labels);

% open new figure
figure

% create plot object
layout_plot = plot(layout_graph);

% set layout method
%layout(layout_plot,'force','Iterations',1000)
%layout(layout_plot,'subspace','Dimension',12)

% alternatively, import layout coordinates generated by Mathematica
coordinates_imported = importdata('E:\Downloads\coordinates.mat');
layout_plot.XData = coordinates_imported(:,1);
layout_plot.YData = coordinates_imported(:,2);

% get node coordinates
layout_spring_coordinates.X = layout_plot.XData;
layout_spring_coordinates.Y = layout_plot.YData;

% save node coordinates
save(fullfile(layout_save_path,'layout_spring_coordinates'),'layout_spring_coordinates');
save(fullfile(layout_save_path,'adjacency_table'),'adjacency_table');
save(fullfile(layout_save_path,'channel_labels'),'channel_labels');


%% plot evolution of nodes out-degree across sessions
% requires: network_adj_matrices, channel_labels

%-----decide if to use unweighted network: network_adj_matrices = network_adj_matrices>0;

% find number of nodes
network_size = size(network_adj_matrices,1);

% compute out-degree of nodes for all sessions
network_outdegree = squeeze(mean(network_adj_matrices,2));

% plot outdegree vs. sessions
figure
plot(network_outdegree')
legend(channel_labels,'Location','northwest')

% smoothen data using sliding average
sliding_window_size = 3;
network_outdegree_smooth = NaN([network_size,sessions]);
for node=1:network_size
    network_outdegree_smooth(node,:) = smooth(network_outdegree(node,:),sliding_window_size);
end
figure
plot(network_outdegree_smooth')
legend(channel_labels,'Location','northwest')

%% plot evolution of nodes in-degree across sessions
% requires: network_adj_matrices, channel_labels

% find number of nodes
network_size = size(network_adj_matrices,1);

% compute in-degree of nodes for all sessions
network_indegree = squeeze(mean(network_adj_matrices,1));

% plot outdegree vs. sessions
figure
plot(network_indegree')
legend(channel_labels,'Location','northwest')

% smoothen data using sliding average
sliding_window_size = 3;
network_indegree_smooth = NaN([network_size,sessions]);
for node=1:network_size
    network_indegree_smooth(node,:) = smooth(network_indegree(node,:),sliding_window_size);
end
figure
plot(network_indegree_smooth')
legend(channel_labels,'Location','northwest')

%% plot network (single session)
% requires: network_adj_matrices, channel_labels, layout_spring_coordinates

network_outdegree = squeeze(mean(network_adj_matrices,2));

session = 8;

% create directed graph object
G = digraph(network_adj_matrices(:,:,session),channel_labels);

% open new figure
figure
% create plot object
network_plot = plot(G);
% set node coordinates
network_plot.XData = layout_spring_coordinates.X;
network_plot.YData = layout_spring_coordinates.Y;
% set node size
network_plot.MarkerSize = 4;
% define node colors according to average outcoming link weight
G.Nodes.NodeColors = network_outdegree(:,session);
network_plot.NodeCData = G.Nodes.NodeColors;
colormap jet
colorbar
% set links properties
network_plot.EdgeColor = 'black';
%G.Edges.LWidths = 4*G.Edges.Weight/max(G.Edges.Weight);
%network_plot.LineWidth = G.Edges.LWidths;
[temp_sources,temp_targets] = find(network_adj_matrices(:,:,session)>=0.5);
highlight(network_plot,temp_sources,temp_targets,'EdgeColor','r','LineWidth',2)


%% generate frames for video

% Specify PNG images folder
PNG_Folder = 'C:\Users\Leo\Desktop\test video';

figure
for network_time_steps = 1:length(results.parameters.samples_timesteps)
    
    G = network_threshold(results.parameters.channel_pairs,results.p_table,network_time_steps,network_p_threshold);
    
    network_plot = plot(G);
    network_plot.MarkerSize = 6;
    network_plot.NodeColor = 'black';
    G.Nodes.NodeColors = outdegree(G);
    network_plot.NodeCData = G.Nodes.NodeColors;
    % colorbar
    network_plot.EdgeColor = 'black';
    
    % define layout or specify nodes coordinates
    %layout(network_plot,'force')
    network_plot.XData = [0.139546361445738 -0.980997368470044 0.900811272096910 -0.870868353268244 0.938937967136015 -0.151116602442996 0.111612190032202 0.879307854870773 -1.75207024939917 0.268360907532298 -1.36321577444384 1.87969179491037];
    network_plot.YData = [-0.826009629651042 1.12572184984103 1.68974504456952 -0.262137067541721 -0.386923724050505 1.30007889896241 0.423620020380977 0.993689257024401 -0.553505599646456 -2.47037314620060 -2.03347165630942 0.499565752621404];
    
    fig = gcf;
    fig.PaperUnits = 'centimeters';
    fig.PaperPosition = [0 0 30 15];
    fig.PaperPositionMode = 'manual';
    print(fullfile(PNG_Folder,sprintf('ScreenSizeFigure%0.5d',network_time_steps)),'-dpng','-r0')
    
end

%% Make an AVI video from a collection of PNG images in a folder.

video_title = 'network.avi';
video_frame_rate = 2;
video_quality = 100;

video_from_png(PNG_Folder,video_title,video_frame_rate,video_quality)


%% reconstruct adjacency matrix from graph object
%[sources,targets] = findedge(G);
%adj = full(sparse(sources,targets,G.Edges.Weight,numnodes(G),numnodes(G)));


