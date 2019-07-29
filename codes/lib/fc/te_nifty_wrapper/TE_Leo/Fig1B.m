% Fig.1B (outdegree, indegree and centrality)

% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m


% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
pagerank_constant = 0.85;

% import channel labels
[FileName,PathName] = uigetfile('*.mat','Select the channel labels file (*.mat)');
load(fullfile(PathName,FileName))
% select session folders
%fprintf('Select session folders\n');[path_names] = uigetfile_n_dir;
[path_names] = uigetfile_n_dir('','Select session folders');

sessions = length(path_names);

% import performance files

% check that every folder contains performance file
for count = 1:sessions
    path_session_id = path_names{1,count};
    if ~exist(fullfile(path_session_id,'Transfer Entropy','performance.mat'),'file')
        error('ERROR: performance.mat file missing in folder: %s',fullfile(path_session_id,'Transfer Entropy'))
    end
end
% loop through selected session folders and import performance files
performance = NaN([1,sessions]);
for count = 1:sessions
    % load results file
    path_session_id = path_names{1,count};
    temp = load(fullfile(path_session_id,'Transfer Entropy','performance.mat'));
    performance(count) = temp.performance;
    clear temp
end

% create figures
figure_outdegree = figure;
figure_outdegree.Name = 'Outdegree';
figure_indegree = figure;
figure_indegree.Name = 'Indegree';
figure_pagerank = figure;
figure_pagerank.Name = 'PageRank';

subplot_count = 0;

% repeat plot for CUE, TEX and LIK
% time intervals (in seconds)
% CUE=[1,1.5], TEX=[2,3.5], LIK=[3.5,6]
for network_time_interval = {[1,1.5],[2,3.5],[3.5,6]}
    
    % generate adjacency matrices for GO and NOGO
    
    clear network_adj_matrices_GO
    clear network_adj_matrices_NOGO
    
    for trials_type = {'GO','NOGO'};
        
        % check that every folder contains results file
        for count = 1:sessions
            path_session_id = path_names{1,count};
            if ~exist(fullfile(path_session_id,'Transfer Entropy',['results_',trials_type{1,1},'.mat']),'file')
                error('ERROR: results_%s.mat file missing in folder: %s',trials_type{1,1},fullfile(path_session_id,'Transfer Entropy'))
            else
                
                % load results file
                path_session_id = path_names{1,count};
                load(fullfile(path_session_id,'Transfer Entropy',['results_',trials_type{1,1}]),'results')
                
                % select timesteps in the desired time interval
                network_time_steps = find(results.parameters.samples_timesteps >= min(network_time_interval{1,1}) & results.parameters.samples_timesteps < max(network_time_interval{1,1}));
                if length(network_time_steps) < 1
                    error('ERROR: no data points found in the specified time interval.')
                end
                
                % store the new adjacency matrix (assumption: all training sessions have the same number of channels)
                if strcmp(trials_type{1,1},'GO')
                    network_adj_matrices_GO(:,:,count) = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                elseif strcmp(trials_type{1,1},'NOGO')
                    network_adj_matrices_NOGO(:,:,count) = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                end
            end
        end
    end
    
    % find high-performance sessions
    high_perf_sessions = performance > network_performance_threshold;
    high_perf_sessions_indices = find(performance > network_performance_threshold);
    
    subplot_count = subplot_count + 1;
    
    % update outdegree figure
    figure(figure_outdegree)
    subplot(3,1,subplot_count)
    
    % compute outdegree of nodes
    %GO
    network_outdegree_GO = squeeze(sum(network_adj_matrices_GO(:,:,high_perf_sessions)>0,2));
    mean_GO = mean(network_outdegree_GO,2);
    var_GO = std(network_outdegree_GO,0,2);
    %NOGO
    network_outdegree_NOGO = squeeze(sum(network_adj_matrices_NOGO(:,:,high_perf_sessions)>0,2));
    mean_NOGO = mean(network_outdegree_NOGO,2);
    var_NOGO = std(network_outdegree_NOGO,0,2);
    
    % plot outdegree vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('Out-degree')
    set(gca,'XTickLabel',channel_labels,'XTickLabelRotation',45)
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
    network_indegree_GO = squeeze(sum(network_adj_matrices_GO(:,:,high_perf_sessions)>0,1));
    mean_GO = mean(network_indegree_GO,2);
    var_GO = std(network_indegree_GO,0,2);
    %NOGO
    network_indegree_NOGO = squeeze(sum(network_adj_matrices_NOGO(:,:,high_perf_sessions)>0,1));
    mean_NOGO = mean(network_indegree_NOGO,2);
    var_NOGO = std(network_indegree_NOGO,0,2);
    
    % plot indegree vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('In-degree')
    set(gca,'XTickLabel',channel_labels,'XTickLabelRotation',45)
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
    network_pagerank_GO = NaN([size(network_adj_matrices_GO,1),length(high_perf_sessions_indices)]);
    
    for count_sessions = 1:length(high_perf_sessions_indices)
        [largest_eigenvector,~] = eigs(pagerank_constant*network_adj_matrices_GO(:,:,high_perf_sessions_indices(count_sessions))+(1-pagerank_constant)*ones(size(network_adj_matrices_GO,1)),1);
        network_pagerank_GO(:,count_sessions) = largest_eigenvector/sum(largest_eigenvector);
    end
    mean_GO = mean(network_pagerank_GO,2);
    var_GO = var(network_pagerank_GO,0,2);
    % NOGO
    network_pagerank_NOGO = NaN([size(network_adj_matrices_NOGO,1),length(high_perf_sessions_indices)]);
    
    for count_sessions = 1:length(high_perf_sessions_indices)
        [largest_eigenvector,~] = eigs(pagerank_constant*network_adj_matrices_NOGO(:,:,high_perf_sessions_indices(count_sessions))+(1-pagerank_constant)*ones(size(network_adj_matrices_NOGO,1)),1);
        network_pagerank_NOGO(:,count_sessions) = largest_eigenvector/sum(largest_eigenvector);
    end
    mean_NOGO = mean(network_pagerank_NOGO,2);
    var_NOGO = var(network_pagerank_NOGO,0,2);
    
    % plot pagerank vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('PageRank')
    set(gca,'XTickLabel',channel_labels,'XTickLabelRotation',45)
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

