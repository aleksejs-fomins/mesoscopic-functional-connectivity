
%clustering = cat(1,clustering_naive, clustering_expert);

% find the area by label
    
    % get selected channels
    label=[];
    channel_label = {'GP', 'DG', 'VL','Cpu', 'CA1_Py', 'S1_bf','VM', 'LDVL', 'Rt'};
    %channel_label = {'Rt'}
    comb=length(channel_label);
    
    % remove mtp_2 from pooling
    pool_mice=[1 2 3 4 5 7 8];
    mice = length(pool_mice);
    
for i=1:comb
    
    ce_go=[];
    cve_go=[];
    ce_nogo=[];
    cve_nogo=[];
    
    for ii = 1:mice
        
    count_mice=pool_mice(ii);
        
    label(:,count_mice) = strncmp(channel_labels_all{count_mice},channel_label{1,i},3);

    end
    
    [ce_go,cve_go,ce_nogo,cve_nogo]= get_clustering_value(clustering,label,pool_mice);
 %   [cn,cvn]= get_clustering_value(clustering_naive,label,mice);


    subplot(3,3,i)
    
    for ii = 1:mice
    
    count_mice=pool_mice(ii);
    
    errorbarxy([0 1],[ce_nogo(count_mice) ce_go(count_mice)],[0 0],[cve_nogo(count_mice) cve_go(count_mice)]);
    text(1,ce_go(count_mice)+0.0001+0.0001*cve_go(count_mice),mouse_IDs{1,count_mice}(1:end));

    axis square
    axis([0 0.8 0 0.8])
    xlim([-0.5 1.5])
    axis square
    hold on
    
    title(channel_label(i))
    
    end
    
end

%clustering{1,1}.mean.GO(:,2)
hold off

%%

[p,table] =anova1([ce_nogo' ce_go'])

%%
[p,table] = ranksum(ce_nogo', ce_go')
