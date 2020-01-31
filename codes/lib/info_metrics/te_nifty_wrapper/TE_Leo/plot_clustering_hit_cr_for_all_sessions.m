%% --- Pool from all EXPERT sessions clustering coefficient from each mouse ---
%  --- according to a channel label ---

%clustering = cat(1,clustering_naive, clustering_expert);

% find the area by label
    
    % get selected channels
    label=[];
    channel_label = {'GP', 'DG', 'VL','Cpu', 'CA1_Py', 'S1_bf','VM', 'LDVL', 'Rt','M1','CA1_Mol','dCA1'};

    comb=length(channel_label);
    
    c_go=cell(1,comb);
    c_nogo=cell(1,comb);
    
    % remove mtp_2 from pooling
    pool_mice=[1 2 3 4 7];
    mice = length(pool_mice);
    
for i=1:comb
    
    for ii = 1:mice
        
    count_mice=pool_mice(ii);
        
    label(:,count_mice) = strncmp(channel_labels_all{count_mice},channel_label{1,i},5);

    end
    
    [c1,c2]= get_clustering_session(clustering_Hit,clustering_CR,label,pool_mice);
    
    c_go{1,i}=cell2mat(c1);
    c_nogo{1,i}=cell2mat(c2);
    diff_c{1,i}=abs(cell2mat(c1)-cell2mat(c2));
    pool_ch_size(i)=length(cell2mat(c1));
    
end


%% loop across mice and concatenate clustering coefficient
pooling_cat=[];
max_length = max(pool_ch_size);
% concatenate all with NaNs to fill in pooled areas
for i=1:comb
    
   pooling_cat(:,i)= [cell2mat(c_go(1,i)) NaN(max_length -pool_ch_size(i),1)'];
   
end

tmp_mat1= mean(pooling_cat,1,'omitnan');
tmp_mat2= sqrt(var(pooling_cat,[],1,'omitnan'));

groupNames=channel_label;

[~, ind] = sort(tmp_mat1,'descend');
sorted_area = groupNames(1,ind);

barwitherr(tmp_mat2(ind), tmp_mat1(ind),'g')
ylim([0 0.6])
set(gca, 'XTickLabel', sorted_area, 'XTick',1:numel(sorted_area),'TickLabelInterpreter','none')
title('Pooling across mice')
xlabel('Area')
ylabel('Clustering')
%%

statMatrix =pooling_cat(:,ind);
[p,t,stats] = anova1(statMatrix,sorted_area);
results = multcompare(stats,'Dimension',[1 2])

%%
[p,table] =anova1([ce_nogo' ce_go'])

%%
[p,table] = ranksum(ce_nogo', ce_go')