% generate and save network layout from experimental adjacency matrix

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