%% --- Pool MEAN clustering coefficient from each mouse ---
%  --- according to a channel label ---

%clustering = cat(1,clustering_naive, clustering_expert);

% find the area by label
    
    % get selected channels
    label=[];
    channel_label = {'GP', 'DG', 'VL','Cpu', 'CA1_Py', 'S1_bf','VM', 'LDVL', 'Rt','M1','CA1_Mol','dCA1'};
    %channel_label = {'Rt'}
    comb=length(channel_label);
    
    ce_go=[];
    cve_go=[];
    ce_nogo=[];
    cve_nogo=[];
    
    % remove mtp_2 from pooling
    pool_mice=[1 2 3 4 7 8];
    mice = length(pool_mice);
    
for i=1:comb
    
    for ii = 1:mice
        
    count_mice=pool_mice(ii);
        
    label(:,count_mice) = strncmp(channel_labels_all{count_mice},channel_label{1,i},5);

    end
    
    [ce_go(:,i),cve_go(:,i),ce_nogo(:,i),cve_nogo(:,i)]= get_clustering_value(clustering,label,pool_mice);
 %   [cn,cvn]= get_clustering_value(clustering_naive,label,mice);


    subplot(4,3,i)
    
    for ii = 1:mice
        
    count_mice=pool_mice(ii);
    
    errorbarxy([0 1],[ce_nogo(count_mice,i) ce_go(count_mice,i)],[0 0],[cve_nogo(count_mice,i) cve_go(count_mice,i)]);
    text(1,ce_go(count_mice,i)+0.0001+0.0001*cve_go(count_mice,i),mouse_IDs{1,count_mice}(1:end));

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

%% plot mean sorted clustering coefficient

tmp_diff = abs(ce_go(pool_mice,:) - ce_nogo(pool_mice,:));
tmp_mat1= mean(ce_go(pool_mice,:),1,'omitnan');
tmp_mat2= sqrt(var(ce_go(pool_mice,:),[],1,'omitnan'));

groupNames=channel_label;

[~, ind] = sort(tmp_mat1,'descend');
sorted_area = groupNames(1,ind);

barwitherr(tmp_mat2(ind), tmp_mat1(ind),'g')
ylim([0 0.6])
set(gca, 'XTickLabel', sorted_area, 'XTick',1:numel(sorted_area),'TickLabelInterpreter','none')
title('Pooling across mice')
xlabel('Area')
ylabel('Clustering')

%% --- DO ANOVA1 test on mean clustering coefficient for 'Hit' trials --- 

statMatrix =ce_go(pool_mice,ind);
[p,t,stats] = anova1(statMatrix,sorted_area);
results = multcompare(stats,'Dimension',[1 2])

