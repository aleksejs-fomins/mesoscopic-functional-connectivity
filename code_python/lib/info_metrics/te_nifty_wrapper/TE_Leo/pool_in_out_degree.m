channels = {'GP','CA1_Py','dCA1','CA1_Mol','DG'};
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
    number_of_channels=length(channel_label);
    
    pool_mice=[1 2 3 4 5 6 7 8];
    mice = length(pool_mice);
    
% find a matching channel label across mice
    
for i=1:number_of_channels    
    for ii = 1:mice

        count_mice=pool_mice(ii);

        label(:,count_mice) = strncmp(channel_labels_all{count_mice},channel_label{1,i},5);

    end
end

for links={'out','in'}

    if strcmp(links,'out')
        
        % compute outdegree
        %Hit
        network_outdegree_Hit{1,count_channel}= get_degree_trial_channel(network_outdegree_GO,pool_mice,label,links);

        %CR
        network_outdegree_CR{1,count_channel}= get_degree_trial_channel(network_outdegree_NOGO,pool_mice,label,links);

    elseif strcmp(links,'in')
        
        % compute indegree
        % Hit
        network_indegree_Hit{1,count_channel}= get_degree_trial_channel(network_indegree_GO,pool_mice,label,links);

        %CR
        network_indegree_CR{1,count_channel}= get_degree_trial_channel(network_indegree_NOGO,pool_mice,label,links);

    end
    
end

% end run repeat
%end

% end count channel
end

%%
% compare links over texture presentation
TEX_early= 5:15; % 2 to 3 seconds of a trial time
%TEX_late= 14:16; % 2 to 3 seconds of a trial time

% Out-degree
degree_GP=network_outdegree_Hit{1,1};
degree_CA1=network_outdegree_Hit{1,2};
degree_dCA1=network_outdegree_Hit{1,3};
degree_CA1Mol=network_outdegree_Hit{1,4};
degree_DG=network_outdegree_Hit{1,5};

degree_GP_early= mean(degree_GP(:,TEX_early),2);
%degree_GP_late= mean(degree_GP(:,TEX_late),2);
degree_CA1_early= mean(degree_CA1(:,TEX_early),2);
%degree_CA1_late= mean(degree_CA1(:,TEX_late),2);
degree_dCA1_early= mean(degree_dCA1(:,TEX_early),2);
degree_CA1Mol_early= mean(degree_CA1Mol(:,TEX_early),2);
degree_DG_early= mean(degree_DG(:,TEX_early),2);

statMatrixOut=padcat(degree_GP_early,degree_CA1_early,degree_dCA1_early,degree_CA1Mol_early,degree_DG_early);

%[p,t,stats] = anova1(statMatrix,{'GP','CA1_Py','dCA1','DG'});
%results = multcompare(stats)

tmp_smat=mean(statMatrixOut,1,'omitnan');
tmp_smat2 = sqrt(var(statMatrixOut,[],1,'omitnan'));


subplot(2,1,1)
barwitherr([tmp_out_smat2], [tmp_out_smat])
ylim([0 6]);
set(gca, 'XTickLabel', channels)
%title('In degree')
xlabel('Area')
ylabel('Out degree')

% In-degree
degree_GP=network_indegree_Hit{1,1};
degree_CA1=network_indegree_Hit{1,2};
degree_dCA1=network_indegree_Hit{1,3};
degree_CA1Mol=network_indegree_Hit{1,4};
degree_DG=network_indegree_Hit{1,5};

degree_GP_early= mean(degree_GP(:,TEX_early),2);
%degree_GP_late= mean(degree_GP(:,TEX_late),2);
degree_CA1_early= mean(degree_CA1(:,TEX_early),2);
%degree_CA1_late= mean(degree_CA1(:,TEX_late),2);
degree_dCA1_early= mean(degree_dCA1(:,TEX_early),2);
degree_CA1Mol_early= mean(degree_CA1Mol(:,TEX_early),2);
degree_DG_early= mean(degree_DG(:,TEX_early),2);

statMatrixIn=padcat(degree_GP_early,degree_CA1_early,degree_dCA1_early,degree_CA1Mol_early,degree_DG_early);

%[p,t,stats] = anova1(statMatrix,{'GP','CA1_Py','dCA1','DG'});
%results = multcompare(stats)

tmp_smat=mean(statMatrixIn,1,'omitnan');
tmp_smat2 = sqrt(var(statMatrixIn,[],1,'omitnan'));

subplot(2,1,2)
barwitherr([tmp_smat2], [tmp_smat])
ylim([0 6]);
set(gca, 'XTickLabel', channels)
%title('In degree')
xlabel('Area')
ylabel('In degree')
%%

[p,t,stats] = anova1(cat(2,statMatrixIn,statMatrixOut),{'GP In','CA1 Py In','dCA1 In','CA1Mol In','DG In','GP Out','CA1 Py Out','dCA1 Out','CA1Mol Out','DG Out'});
results = multcompare(stats)

