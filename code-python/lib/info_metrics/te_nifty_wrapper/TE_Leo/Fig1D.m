cd 'E:\data\texture_discrimination\mtp_12\mtp_12_TE'

% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
pagerank_constant = 0.85;

% select mice folders
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
    
    network_outdegree_GO = NaN([length(channel_labels_all),mice]);
    network_outdegree_NOGO = NaN([length(channel_labels_all),mice]);
    
    network_indegree_GO = NaN([length(channel_labels_all),mice]);
    network_indegree_NOGO = NaN([length(channel_labels_all),mice]);
    
    network_pagerank_GO = NaN([length(channel_labels_all),mice]);
    network_pagerank_NOGO = NaN([length(channel_labels_all),mice]);
    
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
        
        % find high performance sessions
        high_perf_sessions_indices = find(performance > network_performance_threshold);
        
        % get adjacency matrices for GO and NOGO
        network_adj_matrices_GO = NaN([length(channel_labels_all),length(channel_labels_all),length(high_perf_sessions_indices)]);
        network_adj_matrices_NOGO = NaN([length(channel_labels_all),length(channel_labels_all),length(high_perf_sessions_indices)]);
        for trials_type = {'GO','NOGO'};
            
            temp_pagerank_GO = NaN([length(channel_labels_all),length(high_perf_sessions_indices)]);
            temp_pagerank_NOGO = NaN([length(channel_labels_all),length(high_perf_sessions_indices)]);
            
            for count_sessions = 1:length(high_perf_sessions_indices)
                
                % check if folder contains results file
                path_session_id = fullfile(paths_mice{count_mice},sessions_IDs{1,high_perf_sessions_indices(count_sessions)});
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
                        network_adj_matrices_GO(:,:,count_sessions) = temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice});
                        [largest_eigenvector,~] = eigs(pagerank_constant*network_adj_matrices_GO(:,:,count_sessions)+(1-pagerank_constant)*ones(length(channel_labels_all)),1);
                        temp_pagerank_GO(:,count_sessions) = largest_eigenvector/sum(largest_eigenvector);
                    elseif strcmp(trials_type{1,1},'NOGO')
                        temp_matrix = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                        network_adj_matrices_NOGO(:,:,count_sessions) = temp_matrix(channel_labels_all_indices{count_mice},channel_labels_all_indices{count_mice});
                        [largest_eigenvector,~] = eigs(pagerank_constant*network_adj_matrices_NOGO(:,:,count_sessions)+(1-pagerank_constant)*ones(length(channel_labels_all)),1);
                        temp_pagerank_NOGO(:,count_sessions) = largest_eigenvector/sum(largest_eigenvector);
                    end
                end
                
            end
            
            % compute measures
            if strcmp(trials_type{1,1},'GO')
                network_outdegree_GO(:,count_mice) = squeeze(mean(sum(network_adj_matrices_GO>0, 2), 3));
                network_indegree_GO(:,count_mice) = squeeze(mean(sum(network_adj_matrices_GO>0, 1), 3));
                network_pagerank_GO(:,count_mice) = squeeze(mean(temp_pagerank_GO, 2));
            elseif strcmp(trials_type{1,1},'NOGO')
                network_outdegree_NOGO(:,count_mice) = squeeze(mean(sum(network_adj_matrices_NOGO>0, 2), 3));
                network_indegree_NOGO(:,count_mice) = squeeze(mean(sum(network_adj_matrices_NOGO>0, 1), 3));
                network_pagerank_NOGO(:,count_mice) = squeeze(mean(temp_pagerank_NOGO, 2));
            end         
            
        end
        
    end
      
    subplot_count = subplot_count + 1;
    
    % update outdegree figure
    figure(figure_outdegree)
    subplot(3,1,subplot_count)
    
    %GO
    mean_GO = mean(network_outdegree_GO,2);
    var_GO = std(network_outdegree_GO,0,2);
    %NOGO
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
    
    %GO
    mean_GO = mean(network_indegree_GO,2);
    var_GO = std(network_indegree_GO,0,2);
    %NOGO
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
    
    % GO
    mean_GO = mean(network_pagerank_GO,2);
    var_GO = var(network_pagerank_GO,0,2);
    % NOGO
    mean_NOGO = mean(network_pagerank_NOGO,2);
    var_NOGO = var(network_pagerank_NOGO,0,2);
        
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
    
    % save data for Cue, Texture and Lick phases
    analysis.indegree.GO(:,:,subplot_count) =network_indegree_GO;
    analysis.indegree.NOGO(:,:,subplot_count) =network_indegree_NOGO;
    
    analysis.outdegree.GO(:,:,subplot_count) =network_outdegree_GO;
    analysis.outdegree.NOGO(:,:,subplot_count) =network_outdegree_NOGO;
    
    analysis.pagerank.GO(:,:,subplot_count) = network_pagerank_GO;
    analysis.pagerank.NOGO(:,:,subplot_count) = network_pagerank_NOGO;
    
end

save('analysis.mat','analysis');
