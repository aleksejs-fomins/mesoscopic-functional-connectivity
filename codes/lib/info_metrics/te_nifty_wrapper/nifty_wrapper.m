%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HOW TO USE NIfTy raw
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% StatesRaster ::: [nChannel, nTimeBin, nTrial]
% StatesRaster ::: only natural numbers discretizing x-axis
% VariableIDs  ::: 
% Method = 'JointMI2';
% MII1E2Vals(iT) = instinfo(StatesRaster, Method, VariableIDs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NOTE: core_path variable defined externally

% pwdrez = core_path + "data/sim-ds-mat/";
% addpath(char(core_path + "codes/lib/nifty_wrapper/"));
addpath(char(nifty_path + "NIfTy_Ver1/"));
addpath(char(nifty_path + "TE_Leo/"));

% Get Yaro-like data [nTrial, nTimeBin, nChannel]
% data = rand(param.trials_total, param.timebins_total, param.channels_total);
% load(pwdrez + "source_selftest_rand.mat")
load(source_file_name)

% % Define channels and labels
% param.trials_total   = 200;
% param.timebins_total = 50;
% param.channels_total = 12;  % Number of channels

[param.trials_total, param.timebins_total, param.channels_total] = size(data);
% param.channel_labels = {'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12'};

for i = 1:param.channels_total
  param.channel_labels{i} = ['A' num2str(i)];
end

% Define channel properties
param.channels = 1:param.channels_total;
param.states_method = 'UniCB';
param.states_bins_count = estimate_TE_bins(data);
param.channel_pairs = [nchoosek(param.channels,2);nchoosek(flip(param.channels),2)];

% Define TE estimator properties
param.delay_min = 1;
param.delay_max = 6;
param.delay_stepsize = 1;
param.receiver_embedding_tau = 1;

results = TE_NIfTy(data, param);

results.TE_table = cell_to_array(results.TE_table, 1);
results.p_table = cell_to_array(results.p_table, 1);
results.delay_table = cell_to_array(results.delay_table, 1);
%results.entropy = cell_to_array(results.entropy, 0);

%save(pwdrez + "results_selftest_rand.mat", "results")
save(result_file_name, "results")
