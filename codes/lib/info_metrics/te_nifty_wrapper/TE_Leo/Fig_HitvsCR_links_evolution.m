cd 'G:\mtp_TE'

% Fig. ??

% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m


% define thresholds
network_p_threshold = 0.01;
network_performance_threshold = 0.8;
network_frequency_threshold = 0.5;

% select session folders
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
    
    %subplot_count = subplot_count + 1;
    %subplot(3,1,subplot_count)
    
    
    
    % evolution of Hit vs CR links between consecutive training sessions
    
    % GO
    % find number of nodes and links
    network_size_go = size(network_adj_matrices_GO,1);
    network_links_total_go = network_size_go^2 - network_size_go;
    % find indices of weights above threshold
    links_indices_go = network_adj_matrices_GO > 0;
    % compute number of links
    network_links_go = squeeze(sum(sum(links_indices_go,1),2));
    
    % NOGO
    % find number of nodes and links
    network_size_nogo = size(network_adj_matrices_NOGO,1);
    network_links_total_nogo = network_size_nogo^2 - network_size_nogo;
    % find indices of weights above threshold
    links_indices_nogo = network_adj_matrices_NOGO > 0;
    % compute number of links
    network_links_nogo = squeeze(sum(sum(links_indices_nogo,1),2));
    
    
    
    % compute number of links present in Hit trials
    indices_sum = links_indices_go(:,:,1:end) - links_indices_nogo(:,:,1:end);
    % --- exclusive links for Hit ---
    network_links_hit = squeeze(sum(sum(indices_sum == 1,1),2));
    % compute p-value of shared links
    network_links_hit_p_value = zeros([length(network_links_hit),1]);
%     warning('off','MATLAB:nchoosek:LargeCoefficient');
%     network_links_total = network_links_total_go;
%     for count = 1:sessions-1
%         network_links1 = network_links_hit(count);
%         network_links2 = network_links_hit(count+1);
%         network_links_hit_p_value(count) = nchoosek(network_links_total,network_links_hit(count))*nchoosek(network_links_total-network_links_hit(count),network_links1-network_links_hit(count))*nchoosek(network_links_total-network_links1,network_links2-network_links_hit(count))/(nchoosek(network_links_total,network_links1)*nchoosek(network_links_total,network_links2));
%     end
    
    % plot shared links and total links numbers across sessions
    figure
    plot_links = plot(1:sessions,network_links_go(1:end),':bo',1:sessions,network_links_hit,':r');
    plot_links(1).MarkerFaceColor = 'b';
    plot_links(1).MarkerEdgeColor = 'k';
    xlabel('Training sessions')
    ylabel('Links')
    ylim([0,network_links_total_go])
    legend('Total','Hit trials')
    % overlay markers with color coding for p-values
    marker_color = NaN([length(network_links_hit_p_value),1]);
    marker_color(network_links_hit_p_value >= 0.05) = 3;
    marker_color(network_links_hit_p_value < 0.05) = 2;
    marker_color(network_links_hit_p_value < 0.01) = 1;
    hold on
    scatter_links = scatter(1:sessions,network_links_hit,40,marker_color,'filled');
    scatter_links.MarkerEdgeColor = 'k';
    colormap([0,1,0;1,1,0;1,0,0])
    
    
    % --- exclusive links for CR ---
    network_links_cr = squeeze(sum(sum(indices_sum == -1,1),2));
    % compute p-value of shared links
    network_links_cr_p_value = zeros([length(network_links_cr),1]);
    warning('off','MATLAB:nchoosek:LargeCoefficient');
    network_links_total = network_links_total_nogo;
%     for count = 1:sessions-1
%         network_links1 = network_links_cr(count);
%         network_links2 = network_links_cr(count+1);
%         network_links_hit_p_value(count) = nchoosek(network_links_total,network_links_cr(count))*nchoosek(network_links_total-network_links_cr(count),network_links1-network_links_cr(count))*nchoosek(network_links_total-network_links1,network_links2-network_links_cr(count))/(nchoosek(network_links_total,network_links1)*nchoosek(network_links_total,network_links2));
%     end

    
    % plot shared links and total links numbers across sessions
    figure
    plot_links = plot(1:sessions,network_links_nogo(1:end),':bo',1:sessions,network_links_cr,':r');
    plot_links(1).MarkerFaceColor = 'b';
    plot_links(1).MarkerEdgeColor = 'k';
    xlabel('Training sessions')
    ylabel('Links')
    ylim([0,network_links_total])
    legend('Total','Shared with previous session')
    % overlay markers with color coding for p-values
    marker_color = NaN([length(network_links_cr_p_value),1]);
    marker_color(network_links_cr_p_value >= 0.05) = 3;
    marker_color(network_links_cr_p_value < 0.05) = 2;
    marker_color(network_links_cr_p_value < 0.01) = 1;
    hold on
    scatter_links = scatter(1:sessions,network_links_cr,40,marker_color,'filled');
    scatter_links.MarkerEdgeColor = 'k';
    colormap([0,1,0;1,1,0;1,0,0])
    
end

