% Fig.1A
cd 'E:\data\texture_discrimination\mtp_12\mtp_12_TE'
% add folders containing required scripts to the current MATLAB path
% required scripts:
%   - uigetfile_n_dir.m
%   - network_threshold.m

% define thresholds
PNG_Folder = 'E:\fig\network\video';

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

writerObj = VideoWriter('network_AVI.avi');
writerObj.FrameRate=2;

open(writerObj);

fig = figure;
% reposition M1 node 
layout_spring_coordinates.X(1,8)=2.8162;
layout_spring_coordinates.Y(1,8)=1.0966;

for network_time_steps = 1:50%length(results.parameters.samples_timesteps)
% time intervals (in seconds)
% CUE=[1,1.5], TEX=[2,3.5], LIK=[3.5,6]
%for network_time_interval = {[2,3.5]}
    
    clear network_adj_matrices_GO
    clear network_adj_matrices_NOGO
    
    for trials_type = {'GO','NOGO'}
        
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
%                 network_time_steps = find(results.parameters.samples_timesteps >= min(network_time_interval{1,1}) & results.parameters.samples_timesteps < max(network_time_interval{1,1}));
%                 if length(network_time_steps) < 1
%                     error('ERROR: no data points found in the specified time interval.')
%                 end
                
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

    
%subplot_count = subplot_count + 1;
results.parameters.samples_timesteps
    if results.parameters.samples_timesteps(network_time_steps)<1
        suptitle(['Trial start ' num2str(results.parameters.samples_timesteps(network_time_steps))]);
    elseif and(results.parameters.samples_timesteps(network_time_steps)>1,...
            results.parameters.samples_timesteps(network_time_steps)<2.5)
        suptitle(['Auditory cue ' num2str(results.parameters.samples_timesteps(network_time_steps))]);
    elseif and(results.parameters.samples_timesteps(network_time_steps)>2.5,...
            results.parameters.samples_timesteps(network_time_steps)<3.5)
        suptitle(['Texture presentation ' num2str(results.parameters.samples_timesteps(network_time_steps))]);
    elseif and(results.parameters.samples_timesteps(network_time_steps)>3.5,...
            results.parameters.samples_timesteps(network_time_steps)<5.5)
        suptitle(['Lick ' num2str(results.parameters.samples_timesteps(network_time_steps))]);
    elseif and(results.parameters.samples_timesteps(network_time_steps)>5.5,...
            results.parameters.samples_timesteps(network_time_steps)<7)
        suptitle(['Reward ' num2str(results.parameters.samples_timesteps(network_time_steps))]);
    elseif results.parameters.samples_timesteps(network_time_steps)>7
        suptitle(['Trial End' num2str(results.parameters.samples_timesteps(network_time_steps))]);
    end

    subplot(2,1,1)


    % create plot object for GO
    network_plot = plot(digraph(shared_subnetwork_high_perf_GO,channel_labels),'LineWidth',1,'EdgeColor',[0.3 0.3 0.3]);
    hold on
    % set node coordinates
    network_plot.XData = layout_spring_coordinates.X;
    network_plot.YData = layout_spring_coordinates.Y;
    
    % highlight unique links
    %diff_subnetwork_high_perf =[];
    diff_subnetwork_high_perf = (shared_subnetwork_high_perf_GO - shared_subnetwork_high_perf_NOGO)>0;

    network_plot = plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','b','LineWidth',2,'XData', layout_spring_coordinates.X,'YData', layout_spring_coordinates.Y);
    text(0,2.2,'Hit- trial')
        
    subplot(2,1,2)
    
    % create plot object for NOGO
    network_plot = plot(digraph(shared_subnetwork_high_perf_NOGO,channel_labels),'LineWidth',1,'EdgeColor',[0.3 0.3 0.3]);

    % set node coordinates
    network_plot.XData = layout_spring_coordinates.X;
    network_plot.YData = layout_spring_coordinates.Y;
    
    % highlight unique links
    %diff_subnetwork_high_perf =[];
    diff_subnetwork_high_perf = (shared_subnetwork_high_perf_GO - shared_subnetwork_high_perf_NOGO)<0;
    
    plot(digraph(diff_subnetwork_high_perf,channel_labels),'EdgeColor','r','LineWidth',2,'XData', layout_spring_coordinates.X,'YData', layout_spring_coordinates.Y);
    text(0,2.2,'CR- trial')
    
%     fig = gcf;
%     fig.PaperUnits = 'centimeters';
%     fig.PaperPosition = [0 0 30 15];
%     fig.PaperPositionMode = 'manual';
  print(fullfile(PNG_Folder,sprintf('figure%0.5d',network_time_steps)),'-dpng','-r0')
    
  filename = fullfile(PNG_Folder,sprintf('figure%0.5d',network_time_steps));
  thisimage = getframe(gcf);
  writeVideo(writerObj, thisimage);

end

close(writerObj);
