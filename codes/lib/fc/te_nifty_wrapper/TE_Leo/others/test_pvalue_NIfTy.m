%%

temp_trials_number = 100;
temp_states_number = 4;

x = randi(temp_states_number,1,temp_trials_number);
yPast = randi(temp_states_number,1,temp_trials_number);
yFuture = randi(temp_states_number,1,temp_trials_number);

%%

% Error check the integer states
if ~isequal(unique(x),(1:length(unique(x))))
    [B,I,x] = unique(x);
end
if ~isequal(unique(yPast),(1:length(unique(yPast))))
    [B,I,yPast] = unique(yPast);
end
if ~isequal(unique(yFuture),(1:length(unique(yFuture))))
    [B,I,yFuture] = unique(yFuture);
end

%%

% Make a counts matrix
Counts = accumarray({yFuture,yPast,x},ones(size(x)),[length(unique(yFuture)),length(unique(yPast)),length(unique(x))]);

% Calculate the pairwise mutual information
InfoVal = TE2(Counts);


%% Perform the Monte Carlo Trials

MCnSamples = 5000;
MCpThresh = 0.001;

% Figure out how many failures we're allowed
nFailsThresh = ceil(MCpThresh * MCnSamples);
%nFailsThresh = 5000;
iSample = 1;
iFails = 0;
MCInfoVals = NaN([1,MCnSamples]);
nyFuture = length(unique(yFuture));
nyPast = length(unique(yPast));
nx = length(unique(x));
lx = length(x);

while (iFails < nFailsThresh) && (iSample <= MCnSamples)
    % Counts = accumarray({yFuture,yPast,x(randperm(lx))},ones(size(x)),[nyFuture,nyPast,nx]);
    MCInfoVals(iSample) = TE2(accumarray({yFuture,yPast,x(randperm(lx))},ones([lx,1]),[nyFuture,nyPast,nx]));
    iFails(MCInfoVals(iSample) >= InfoVal) = iFails(MCInfoVals(iSample) >= InfoVal) + 1;
    iSample = iSample + 1;
end

% Calculate the p-value
p = iFails / (iSample - 1);

% Correct for the resolution of the Monte Carlo trials
%p(p == 0) = 1/(2*MCnSamples);