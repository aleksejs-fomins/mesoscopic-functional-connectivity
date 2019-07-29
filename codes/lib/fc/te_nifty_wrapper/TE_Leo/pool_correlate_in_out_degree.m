channels = {'GP','DG'};
number_of_channels=length(channels);

% calculate number of links
network_outdegree_Hit=[];
network_outdegree_CR=[];
network_indegree_Hit=[];
network_indegree_CR=[];

for count_channel=1:number_of_channels
    
%for run_repeat=1:10

% get selected channels
    label=[];
    channel_label = channels(count_channel);
    number_of_channels=length(channels);
    
    pool_mice=[1 2 3 4 5 6 7 8];
    mice = length(pool_mice);
    
% find a matching channel label across mice
choose_pool=[];
count_mice=1;
for ii = 1:mice
    
    for i=1:number_of_channels    

        count_mice=pool_mice(ii);

        label_control(:,count_mice,i) = strncmp(channel_labels_all{count_mice},channels{1,i},5);
    end
    
    % choose mice to pool which have both channel labels
    if sum(sum(squeeze(label_control(:,count_mice,:)),1),2)==2
        choose_pool=[choose_pool pool_mice(ii)];
        label(:,count_mice)= strncmp(channel_labels_all{count_mice},channel_label,5);
        count_mice= count_mice+1;
    else
        
    end
    
end


for links={'out','in'}

    if strcmp(links,'out')
        
        % compute outdegree
        %Hit
        network_outdegree_Hit{1,count_channel}= get_degree_trial_channel(network_outdegree_GO,choose_pool,label,links);

        %CR
        network_outdegree_CR{1,count_channel}= get_degree_trial_channel(network_outdegree_NOGO,choose_pool,label,links);

    elseif strcmp(links,'in')
        
        % compute indegree
        % Hit
        network_indegree_Hit{1,count_channel}= get_degree_trial_channel(network_indegree_GO,choose_pool,label,links);

        %CR
        network_indegree_CR{1,count_channel}= get_degree_trial_channel(network_indegree_NOGO,choose_pool,label,links);

    end
    
end

end

% correlate consecutive channels

% Build the filter
times = 2;
sigma = 1;
size2 = 20;
x = linspace(-size2 / 2, size2 / 2, size2);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter);


%% Arrange out degree in a vector
%network_outdegree_Hit
% {}{}{}
time_interval={1:10,10:15,15:20};

cross.coeff=[];
cross.lag=[];

for count_time_interval = 1:length(time_interval)

network_time_interval=time_interval{count_time_interval};

% --- to shuffle time samples --- introduce time_shuffle to
% --- network_time_interval(time_shuffle) in n_GP and n_DG---
time_shuffled=randperm(length(network_time_interval));

% for GP
count_channel=1;
n_GP = network_outdegree_Hit{1,count_channel}(:,network_time_interval);
% N_GP = n_GP(:,:)';
% N_GP = N_GP(:)';
meanN_GP = mean(N_GP);
stdN_GP = std(N_GP);

% for DG
count_channel=2;
n_DG = network_outdegree_Hit{1,count_channel}(:,network_time_interval);
% N_DG = n_DG(:,:)';
% N_DG = N_DG(:)';
meanN_DG = mean(N_DG);
stdN_DG = std(N_DG);

[sessions,~]=size(n_GP);
maxlag= 10;
    
    for count_session = 1:sessions
                
                % maxlag limits lags calculation
                [xc,lgs] = xcorr(n_GP(count_session,:),n_DG(count_session,:),maxlag,'coeff');
                %[cr,p_values] = corrcoef(data_var(j,:,i),behavior_var(j,:));
                
                %[M,I] = max(cr(round(length(cr)/2 - delta): round(length(cr)/2 + delta)));
                [M,I] = max(abs(xc));
                
                % if lagDiff is "-" then calcium is earlier then whisking
                % if lagDiff is "+" then calcium is later then whisking
                % lagDiff in seconds

                lagDiff(count_session) = lgs(I)*200e-3;
                
                % before using corrcoeff cc equal to M
                % corrcoef is normalized to the cc at a zero lag (signal's energy)
                cc(count_session) = M;

    end

    cross.coeff = cat(1,cross.coeff,cc);
    cross.lag = cat(1,cross.lag,lagDiff);


    cross.mean_coeff(count_time_interval) = mean(cc,2,'omitnan');
    cross.var_coeff(count_time_interval) = var(cc,[],2,'omitnan');
    % 
    cross.mean_lag(count_time_interval) = mean(lagDiff,2,'omitnan');
    cross.var_lag(count_time_interval) = var(lagDiff,[],2,'omitnan');
    
end


%%
%[h,p]=ttest(cross.lag(1,:),cross.lag(2,:))

%[p,t,stats] = anova1(cross.lag',{'Cue','Early Tex','Late Tex'});
[p,t,stats] = anova1(cross.lag',{'Cue','Early Tex','Late Tex'});

results = multcompare(stats)

%% Calculating real Correlation & Bootstrap
Corr = conv(xcorr(N_GP-meanN_GP,N_DG-meanN_DG,5)/stdN_GP/stdN_DG/length(TEX),gaussFilter,'same');
cor_vect= xcorr(N_GP-meanN_GP,N_DG-meanN_DG,5)/stdN_GP/stdN_DG/length(TEX);
plot(cor_vect);

% allCorrW = [];
% 
% for i =1:times
% ff = randperm(length(allfields));
% tmpN = aN(ff,:);
% tmpN = tmpN';
% tmpN = tmpN(:)';
% 
% tmpW = aW(ff,:);
% tmpW = tmpW';
% tmpW = tmpW(:)';
% 
% allCorrW(i,:) = conv(xcorr(tmpN-meanN,tmpW-meanW,300)/stdN/stdW/601,gaussFilter,'same');
% end

% if(Corr.W.valid==1 && size(W,2)~=1); Corr.W.CI = prctile(allCorrW,[100*0.05/2,100*(1-0.05/2)]); end



