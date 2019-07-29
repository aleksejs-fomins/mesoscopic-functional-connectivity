% create figures
figure_outdegree = figure;
figure_outdegree.Name = 'Outdegree';
figure_indegree = figure;
figure_indegree.Name = 'Indegree';
figure_clustering = figure;
figure_clustering.Name = 'Clustering';
figure_closeness = figure;
figure_closeness.Name = 'Closeness';

% plot only for the texture presentation time

subplot_count =2;

for mice_count=1:length(mouse_IDs)
    
    channel_labels = channel_labels_all{mice_count};
    %------------------------------    
    %--- update outdegree figure ---
    %------------------------------
% 
    figure(figure_outdegree)
    subplot(length(mouse_IDs)/2,2,mice_count)
    
    %Initialize data with zeros
    mean_GO = zeros(12,1);
    mean_NOGO = zeros(12,1);
    
    var_GO = zeros(12,1);
    var_NOGO = zeros(12,1);
    
    %Outdegree GO/NOGO
    mean_GO = outdegree{1,mice_count}.mean.GO(:,subplot_count);
    mean_NOGO = outdegree{1,mice_count}.mean.NOGO(:,subplot_count);
    
    var_GO = outdegree{1,mice_count}.var.GO(:,subplot_count);
    var_NOGO = outdegree{1,mice_count}.var.NOGO(:,subplot_count);
   
    % plot outdegree vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('Out-degree')
    set(gca,'XTickLabel',channel_labels,'XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO,var_NOGO];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    
    
    %------------------------------    
    %--- update indegree figure ---
    %------------------------------

    figure(figure_indegree)
    subplot(length(mouse_IDs)/2,2,mice_count)
    
    %Indegree GO/NOGO
    
    mean_GO = indegree{1,mice_count}.mean.GO(:,subplot_count);
    mean_NOGO = indegree{1,mice_count}.mean.NOGO(:,subplot_count);
    
    var_GO = indegree{1,mice_count}.var.GO(:,subplot_count);
    var_NOGO = indegree{1,mice_count}.var.NOGO(:,subplot_count);
    
    if or(sum(size(mean_GO)<10),sum(size(mean_NOGO)<10))
        mean_GO = zeros(12,1);
        mean_NOGO = zeros(12,1);

        var_GO = zeros(12,1);
        var_NOGO = zeros(12,1);
    end
    
    % plot indegree vs. channel labels
    temp_data = [mean_GO,mean_NOGO];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('In-degree')
    set(gca,'XTickLabel',channel_labels,'XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO,var_NOGO];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    
    
    %------------------------------
    %--- update clustering figure ---
    %------------------------------

    
    figure(figure_clustering)
    subplot(length(mouse_IDs)/2,2,mice_count)
    
    %Initialize data with zeros
    mean_GO = zeros(12,1);
    mean_NOGO = zeros(12,1);
    
    var_GO = zeros(12,1);
    var_NOGO = zeros(12,1);
    
    % Clustering coefficient GO/NOGO
     
    mean_GO = clustering{1,mice_count}.mean.GO(:,subplot_count);
    mean_NOGO = clustering{1,mice_count}.mean.NOGO(:,subplot_count);
    
    var_GO = clustering{1,mice_count}.var.GO(:,subplot_count);
    var_NOGO = clustering{1,mice_count}.var.NOGO(:,subplot_count);
    
    % sort by mean GO trials
    [sorted, SORTi] = sort(mean_GO);
    for i = 1:length(SORTi)
        channel_sorted{i} = channel_labels{SORTi(i)};
    end
    
    % plot pagerank vs. channel labels
    temp_data = [mean_GO(SORTi), mean_NOGO(SORTi)];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('Clustering')
    set(gca,'XTickLabel',channel_sorted','XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO(SORTi), var_NOGO(SORTi)];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    title(mouse_IDs{mice_count})

    %------------------------------
    %--- update eigencentrality ---
    %------------------------------
    
    figure(figure_closeness)
    subplot(length(mouse_IDs)/2,2,mice_count)
    
    %Initialize data with zeros
    mean_GO = zeros(12,1);
    mean_NOGO = zeros(12,1);
    
    var_GO = zeros(12,1);
    var_NOGO = zeros(12,1);
    
    % Closeness GO/NOGO

    mean_GO = eigencentrality{1,mice_count}.mean.GO(:,subplot_count);
    mean_NOGO = eigencentrality{1,mice_count}.mean.NOGO(:,subplot_count);
    
    var_GO = eigencentrality{1,mice_count}.var.GO(:,subplot_count);
    var_NOGO = eigencentrality{1,mice_count}.var.NOGO(:,subplot_count);
    
    % sort
    [sorted, SORTi] = sort(mean_GO);
    for i = 1:length(SORTi)
        channel_sorted{i} = channel_labels{SORTi(i)};
    end
    
    % plot differences in a network vs. channel labels
    temp_data = [mean_GO(SORTi), mean_NOGO(SORTi)];
    temp_barplot = bar(temp_data);
    ylim([0,inf])
    ylabel('Closeness')
    set(gca,'XTickLabel',channel_sorted','XTickLabelRotation',45)
    legend({'GO','NOGO'},'Location','northeast')
    
    % add errorbars
    temp_errorbars = [var_GO(SORTi), var_NOGO(SORTi)];
    temp_errorbars_x = NaN(size(temp_data));
    temp_errorbars_y = NaN(size(temp_data));
    for temp_bar_count = 1:size(temp_data,2)
        temp_errorbars_x(:,temp_bar_count) = bsxfun(@plus, temp_barplot(1).XData', [temp_barplot(temp_bar_count).XOffset]);
        temp_errorbars_y(:,temp_bar_count) = temp_barplot(temp_bar_count).YData';
    end
    hold on
    errorbar(temp_errorbars_x, temp_errorbars_y, temp_errorbars, '.r')
    hold off
    
end
