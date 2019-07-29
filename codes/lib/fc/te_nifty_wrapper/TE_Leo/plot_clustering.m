
comb = 9;

for i=1:comb
    
    % get selected channels
    label=[];

    channel_label1 = {'DG', 'DG', 'DG', 'CA1_Py', 'CA1_Py', 'CA1_Py','M1','M1','M1'};

    for count_mice = 1:mice

            label.one(:,count_mice) = strncmp(channel_labels_all{count_mice},channel_label1{1,i},3);

    end

    % add a second label
    % for count_mice = 1:mice
    %        
    %         label.one(:,count_mice) = or(strncmp(channel_labels_all{count_mice},channel_label,3), ...
    %             strncmp(channel_labels_all{count_mice},'CA1_Mol',3));
    % 
    % end

    channel_label2 = {'S1_bf', 'VL', 'GP', 'S1_bf', 'VL', 'GP', 'S1_bf', 'VL', 'GP'};

    for count_mice = 1:mice

            label.two(:,count_mice) = strncmp(channel_labels_all{count_mice},channel_label2{1,i},3);

    end

    % add a second label
    % for count_mice = 1:mice
    %        
    %         label.two(:,count_mice) = or(strncmp(channel_labels_all{count_mice},channel_label,3), ...
    %             strncmp(channel_labels_all{count_mice},'LDVL',3));
    % 
    % end
                 

%[c1,cv1,c2,cv2]= find_clusters_labels(clustering,label,mice);
[c1,cv1,c2,cv2]= find_clusters_labels(outdegree,label,mice);

subplot(3,3,i)

for count_mice = 1:mice
        errorbarxy(c1(count_mice),c2(count_mice),cv1(count_mice),cv2(count_mice),{'ko', 'b', 'r'});
    text(c1(count_mice)+0.01+0.1*cv1(count_mice),c2(count_mice)+0.01+0.1*cv2(count_mice),mouse_IDs{1,count_mice});
    hold on
    
    xlabel(channel_label1{1,i})
    ylabel(channel_label2{1,i})
    %superimpose a least-squares line on the scatter plot.
    %lsline
    % refline
    hline = refline([1 0]);
    hline.Color = 'r';
    hline.LineStyle=':';
    axis square
    %axis([0 0.8 0 0.8])
    
end

hold off

title('Clustering coeff')
end


