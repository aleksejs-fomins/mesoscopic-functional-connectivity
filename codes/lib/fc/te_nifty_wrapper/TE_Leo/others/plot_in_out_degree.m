%% indegree  averaged across areas

cue_go = mean(analysis.indegree.GO(:,:,1),1);
cue_std_go = std(analysis.indegree.GO(:,:,1),0,1);

tex_go =  mean(analysis.indegree.GO(:,:,2),1);
tex_std_go =  std(analysis.indegree.GO(:,:,2),0,1);

lik_go =  mean(analysis.indegree.GO(:,:,3),1);
lik_std_go =  std(analysis.indegree.GO(:,:,3),0,1);

% nogo
cue_nogo = mean(analysis.indegree.NOGO(:,:,1),1);
cue_std_nogo = std(analysis.indegree.NOGO(:,:,1),0,1);

tex_nogo =  mean(analysis.indegree.NOGO(:,:,2),1);
tex_std_nogo =  std(analysis.indegree.NOGO(:,:,2),0,1);

lik_nogo =  mean(analysis.indegree.NOGO(:,:,3),1);
lik_std_nogo =  std(analysis.indegree.NOGO(:,:,3),0,1);

%% outdegree averaged across areas

cue_go = mean(analysis.outdegree.GO(:,:,1),1);
cue_std_go = std(analysis.outdegree.GO(:,:,1),0,1);

tex_go =  mean(analysis.outdegree.GO(:,:,2),1);
tex_std_go =  std(analysis.outdegree.GO(:,:,2),0,1);

lik_go =  mean(analysis.outdegree.GO(:,:,3),1);
lik_std_go =  std(analysis.outdegree.GO(:,:,3),0,1);

% nogo
cue_nogo = mean(analysis.outdegree.NOGO(:,:,1),1);
cue_std_nogo = std(analysis.outdegree.NOGO(:,:,1),0,1);

tex_nogo =  mean(analysis.outdegree.NOGO(:,:,2),1);
tex_std_nogo =  std(analysis.outdegree.NOGO(:,:,2),0,1);

lik_nogo =  mean(analysis.outdegree.NOGO(:,:,3),1);
lik_std_nogo =  std(analysis.outdegree.NOGO(:,:,3),0,1);

%% do a bar plot of 
x = [0 1 2];
marker=['o' 's' 'd' '^' 'v'];

for i=1:length(cue_go)
    
    subplot(2,1,1)
    errorbar([0 1 2],[cue_go(i) tex_go(i) lik_go(i)],[cue_std_go(i) tex_std_go(i) lik_std_go(i)],'-',...
    'Marker', marker(i) ,...
    'Color', [0.8, 0.8, 0.8],...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor',[0,0,0.8],...
    'MarkerFaceColor',[0.8,0.8,0.8])
    xlim([-0.5 2.5])
    axis square
    hold on
    
    subplot(2,1,2)
    errorbar([0 1 2],[cue_nogo(i) tex_nogo(i) lik_nogo(i)],[cue_std_nogo(i) tex_std_nogo(i) lik_std_nogo(i)],'-',...
    'Marker', marker(i) ,...
    'Color', [0.8, 0.8, 0.8],...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor',[0.8,0,0],...
    'MarkerFaceColor',[0.8,0.8,0.8])
    xlim([-0.5 2.5])
    axis square
    hold on
end

hold off

%% do anova 

%one sample Kolmogorov-Smirnov test to check if data are normally
%distributed

h = kstest(tex_go);


[p,table] = anova1([lik_go' tex_go']);

%% data are not normal do ranksum

[p_go h_go] = ranksum(lik_go', tex_go');
%[p_nogo h_nogo] = ranksum(cue_nogo', tex_nogo');