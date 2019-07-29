% Fig.1C

% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m
%addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\mtp\');
%addpath('E:\Google Drive\University\UZH HIFO Job Yaroslav Sych\matlab\TE_Leo\');

% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
network_frequency_threshold = 0.5;

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
    
    % select shared links among all high-performance sessions and all mice (shared subnetwork)
    shared_subnetwork_high_perf_GO = all(all(network_adj_matrices_GO(:,:,high_perf_sessions,:),4),3);
    
    %subplot_count = subplot_count + 1;
    %subplot(3,1,subplot_count)
    figure
    
    % create plot object for GO
    network_plot = plot(digraph(shared_subnetwork_high_perf_GO,channel_labels_shared));
    % set node coordinates
    %network_plot.XData = layout_spring_coordinates.X;
    %network_plot.YData = layout_spring_coordinates.Y;
    layout(network_plot,'circle')
    % set links properties
    network_plot.EdgeColor = 'black';
    
    % highlight links above frequency threshold
    hold on
    plot(digraph(mean(mean(network_adj_matrices_GO(:,:,high_perf_sessions,:),4),3)>= network_frequency_threshold,channel_labels_shared),'Marker','none','EdgeColor','r','LineWidth',2,'layout', 'circle');
    plot(digraph(mean(mean(network_adj_matrices_NOGO(:,:,high_perf_sessions,:),4),3)>= network_frequency_threshold,channel_labels_shared),'Marker','none','EdgeColor','b','LineWidth',2,'layout', 'circle');
    
end
