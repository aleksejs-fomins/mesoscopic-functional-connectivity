% Fig.1A
cd 'E:\data\texture_discrimination\mvg_12\mvg_12_TE'
% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m

% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
network_frequency_threshold = 0.5;

% import channel labels
[FileName,PathName] = uigetfile('*.mat','Select the channel labels file (*.mat)');
load(fullfile(PathName,FileName))
% import network spring layout
[FileName,PathName] = uigetfile('*.mat','Select the network layout file (*.mat)');
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

% generate adjacency matrices for GO and NOGO
subplot_count = 0;
% time intervals (in seconds)
% CUE=[1,1.5], TEX=[2,3.5], LIK=[3.5,6]
for network_time_interval = {[2,3.5]}
    
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
                
                % add the new adjacency matrix (assumption: all training sessions have the same number of channels)
                if strcmp(trials_type{1,1},'GO')
                    network_adj_matrices_GO(:,:,count) = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                elseif strcmp(trials_type{1,1},'NOGO')
                    network_adj_matrices_NOGO(:,:,count) = network_threshold(results.p_table,network_time_steps,network_p_threshold);
                end
            end
        end
        
    end
    
    % select shared links among all high-performance sessions (shared subnetwork)
    
    high_perf_sessions = performance > network_performance_threshold;
    
    shared_subnetwork_high_perf_GO = all(network_adj_matrices_GO(:,:,high_perf_sessions),3);
    shared_subnetwork_high_perf_NOGO = all(network_adj_matrices_NOGO(:,:,high_perf_sessions),3);

    
%% %subplot_count = subplot_count + 1;
    subplot(2,1,1)
    
    % create plot object for GO
    network_plot = plot(digraph(shared_subnetwork_high_perf_GO,channel_labels),'EdgeColor',[0.5 0.5 0.5]);
    hold on
    % set node coordinates
    network_plot.XData = layout_spring_coordinates.X;
    network_plot.YData = layout_spring_coordinates.Y;
    
    % highlight unique links
    
    diff_subnetwork_high_perf = (shared_subnetwork_high_perf_GO - shared_subnetwork_high_perf_NOGO)>0;

    network_plot = plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','b','LineWidth',2,'XData', layout_spring_coordinates.X,'YData', layout_spring_coordinates.Y);
    hold off
    
    subplot(2,1,2)
    
    % create plot object for NOGO
    network_plot = plot(digraph(shared_subnetwork_high_perf_NOGO,channel_labels),'EdgeColor',[0.5 0.5 0.5]);
    hold on
    % set node coordinates
    network_plot.XData = layout_spring_coordinates.X;
    network_plot.YData = layout_spring_coordinates.Y;
    
    % highlight unique links
    
    diff_subnetwork_high_perf = (shared_subnetwork_high_perf_GO - shared_subnetwork_high_perf_NOGO)<0;
    
    plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','r','LineWidth',2,'XData', layout_spring_coordinates.X,'YData', layout_spring_coordinates.Y);
    hold off
end

%% plot only Hit and only CR links


subplot(2,1,1)

    diff_subnetwork_high_perf = (shared_subnetwork_high_perf_GO - shared_subnetwork_high_perf_NOGO)>0;

    % create plot object for unique links for Hit trials
    network_plot = plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','r');
    hold on
    % set node coordinates
    network_plot.XData = layout_spring_coordinates.X;
    network_plot.YData = layout_spring_coordinates.Y;
    
    % highlight links above frequency threshold
    plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','b','LineWidth',2,'XData', layout_spring_coordinates.X,'YData', layout_spring_coordinates.Y);
    hold off
    
subplot(2,1,2)

    diff_subnetwork_high_perf = (shared_subnetwork_high_perf_GO - shared_subnetwork_high_perf_NOGO)<0;

    % create plot object for unique links for Hit trials
    network_plot = plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','r');
    hold on
    % set node coordinates
    network_plot.XData = layout_spring_coordinates.X;
    network_plot.YData = layout_spring_coordinates.Y;
    
    % highlight links above frequency threshold
    plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','r','LineWidth',2,'XData', layout_spring_coordinates.X,'YData', layout_spring_coordinates.Y);
    hold off