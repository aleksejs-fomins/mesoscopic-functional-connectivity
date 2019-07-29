% Fig.1D Average Network

% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m

% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;

% select mice folders
%fprintf('Select mice folders\n');[path_names] = uigetfile_n_dir;
[paths_mice] = uigetfile_n_dir('','Select mice folders');
mice = length(paths_mice);

% get mouse ID
mouse_IDs = cell([1,mice]);
for count_mice = 1:mice
    [~,mouse_IDs{count_mice},~] = fileparts(paths_mice{count_mice});
end

% import all channel labels for comparison
channel_labels_all = cell([1,mice]);
for count_mice = 1:mice
    temp_file_name = fullfile(paths_mice{count_mice},'channel_labels.mat');
    if exist(temp_file_name,'file')
        temp = load(temp_file_name);
        channel_labels_all{count_mice} = temp.channel_labels;
    else
        error('ERROR: channel_labels.mat file missing in folder: %s',paths_mice{count_mice})
    end
end

% find shared channel labels among all mice
channel_labels_shared = channel_labels_all{1};
for count_mice = 1:mice
    channel_labels_shared = intersect(channel_labels_shared,channel_labels_all{count_mice},'stable');
end

% find the indices(positions) of the shared labels in each mouse channel labels
channel_labels_all_indices = cell([1,mice]);
for count_mice = 1:mice
    [~,~,channel_labels_all_indices{count_mice}] = intersect(channel_labels_shared,channel_labels_all{count_mice},'stable');
end

% create figures
figure_outdegree = figure;
figure_outdegree.Name = 'Outdegree';
figure_indegree = figure;
figure_indegree.Name = 'Indegree';
figure_pagerank = figure;
figure_pagerank.Name = 'PageRank';

subplot_count = 0;
% time intervals (in seconds)
% CUE=[1,1.5], TEX=[2,3.5], LIK=[3.5,6]
for network_time_interval = {[1,1.5],[2,3.5],[3.5,6]}
    
    for count_mice = 1:mice
        
        % Get a list of all files and folders in the mouse folder
        all_files = dir(paths_mice{count_mice});
        % Remove . and .. folders
        all_files(1:2) = [];
        % Get a logical vector that tells which is a directory
        dirFlags = [all_files.isdir];
        % Extract only those that are directories.
        sessions_IDs = {all_files(dirFlags).name};
        sessions = length(sessions_IDs);
        
        % import performance files and create performance vector
        performance = NaN([1,sessions]);
        for count_sessions = 1:sessions
            % load results file
            path_session_id = fullfile(paths_mice{count_mice},sessions_IDs{1,count_sessions});
            temp_file_name = fullfile(path_session_id,'Transfer Entropy','performance.mat');
            if exist(temp_file_name,'file')
                temp_loaded_data = load(temp_file_name);
                performance(count_sessions) = temp_loaded_data.performance;
            else
                error('ERROR: performance.mat file missing in folder: %s',fullfile(path_session_id,'Transfer Entropy'))
            end
        end
        
        % generate adjacency matrices for GO and NOGO
        clear network_adj_matrices_GO;
        clear network_adj_matrices_NOGO;
        
        for trials_type = {'GO','NOGO'}
            for count_sessions = 1:sessions
                
                % check if folder contains results file
                path_session_id = fullfile(paths_mice{count_mice},sessions_IDs{1,count_sessions});
                temp_file_name = fullfile(path_session_id,'Transfer Entropy',['results_',trials_type{1,1},'.mat']);
                if ~exist(temp_file_name,'file')
                    error('ERROR: results_%s.mat file missing in folder: %s',trials_type{1,1},fullfile(path_session_id,'Transfer Entropy'))
                else
                    % load results file
                    load(temp_file_name,'results')
                    
                    % select timesteps in the desired time interval
                    network_time_steps = find(results.parameters.samples_timesteps >= min(network_time_interval{1,1}) & results.parameters.samples_timesteps < max(network_time_interval{1,1}));
                    if length(network_time_steps) < 1
                        error('ERROR: no data points found in the specified time interval.')
                    end
                    
                    % add the adjacency matrix, only taking the shared channels
                    if strcmp(trials_type{1,1},'GO')
                        temp_matrix = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                        network_adj_matrices_GO(:,:,count_sessions,count_mice) = temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice});
                    elseif strcmp(trials_type{1,1},'NOGO')
                        temp_matrix = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                        network_adj_matrices_NOGO(:,:,count_sessions,count_mice) = temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice});
                    end
                end
                
            end
        end
        
    end
    
    % find high performance sessions
    high_perf_sessions = performance > network_performance_threshold;
    high_perf_sessions_indices = find(performance > network_performance_threshold);
    
    subplot_count = subplot_count + 1;
    
    % update outdegree figure
    figure(figure_outdegree)
    subplot(3,1,subplot_count)
    
    % compute outdegree of nodes
    %GO
    network_outdegree_GO = squeeze(mean(sum(network_adj_matrices_GO(:,:,high_perf_sessions,:)>0, 2), 3));
    mean_GO = mean(network_outdegree_GO,2);
    var_GO = std(network_outdegree_GO,0,2);
    %NOGO
    network_outdegree_NOGO = squeeze(mean(sum(network_adj_matrices_NOGO(:,:,high_perf_sessions,:)>0, 2), 3));
    mean_NOGO = mean(network_outdegree_NOGO,2);
    var_NOGO = std(network_outdegree_NOGO,0,2);
    
    % plot outdegree vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('Out-degree')
    set(gca,'XTickLabel',channel_labels_shared,'XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO,var_NOGO];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    
    
    
    % update indegree figure
    figure(figure_indegree)
    subplot(3,1,subplot_count)
    
    % compute indegree of nodes
    %GO
    network_indegree_GO = mean(sum(network_adj_matrices_GO(:,:,high_perf_sessions,:)>0, 1), 3);
    network_indegree_GO = squeeze(permute(network_indegree_GO,[2 1 3 4]));
    mean_GO = mean(network_indegree_GO,2);
    var_GO = std(network_indegree_GO,0,2);
    %NOGO
    network_indegree_NOGO = mean(sum(network_adj_matrices_NOGO(:,:,high_perf_sessions,:)>0, 1), 3);
    network_indegree_NOGO = squeeze(permute(network_indegree_NOGO,[2 1 3 4]));
    mean_NOGO = mean(network_indegree_NOGO,2);
    var_NOGO = std(network_indegree_NOGO,0,2);
    
    % plot indegree vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('In-degree')
    set(gca,'XTickLabel',channel_labels_shared,'XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO,var_NOGO];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    
    
    
    % update pagerank figure
    figure(figure_pagerank)
    subplot(3,1,subplot_count)
    
    % compute pagerank
    % GO
    network_pagerank_GO = NaN([size(network_adj_matrices_GO,1),length(high_perf_sessions_indices),mice]);
    for count_mice = 1:mice
        for count_sessions = 1:length(high_perf_sessions_indices)
            [largest_eigenvector,~] = eigs(0.85*network_adj_matrices_GO(:,:,high_perf_sessions_indices(count_sessions),count_mice)+0.15*ones(size(network_adj_matrices_GO,1)),1);
            network_pagerank_GO(:,count_sessions,count_mice) = largest_eigenvector/sum(largest_eigenvector);
        end
    end
    mean_GO = mean(mean(network_pagerank_GO,2),3);
    var_GO = var(mean(network_pagerank_GO,2),0,3);
    % NOGO
    network_pagerank_NOGO = NaN([size(network_adj_matrices_NOGO,1),length(high_perf_sessions_indices),mice]);
    for count_mice = 1:mice
        for count_sessions = 1:length(high_perf_sessions_indices)
            [largest_eigenvector,~] = eigs(0.85*network_adj_matrices_NOGO(:,:,high_perf_sessions_indices(count_sessions),count_mice)+0.15*ones(size(network_adj_matrices_NOGO,1)),1);
            network_pagerank_NOGO(:,count_sessions,count_mice) = largest_eigenvector/sum(largest_eigenvector);
        end
    end
    mean_NOGO = mean(mean(network_pagerank_NOGO,2),3);
    var_NOGO = var(mean(network_pagerank_NOGO,2),0,3);
    
    % plot pagerank vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('PageRank')
    set(gca,'XTickLabel',channel_labels_shared,'XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO,var_NOGO];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    
end
