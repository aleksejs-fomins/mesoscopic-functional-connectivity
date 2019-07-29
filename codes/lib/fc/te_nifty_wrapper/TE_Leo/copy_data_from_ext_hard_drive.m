% before running this script and selecting all the session folders of a given mouse in the external
% drive (e.g. all the session folders contained in F:\mtp\mtp_1\), run Yaro's script by typing the
% command MAIN_MTP(0,0) and select all the session folders. If there is an
% error, it usually means some session folders have to be skipped; in this
% case, run MAIN_MTP(0,0) again and skip the bad folders

destination = 'G:\mtp_TE'; %no mouse ID needed

[path_names] = uigetfile_n_dir;

% check that every folder contains data and behavior files
for count = 1:length(path_names)
    
    path_session_id = path_names{1,count};
    
    [path_mouse,session,~] = fileparts(path_session_id);
    [~,mouse_id,~] = fileparts(path_mouse);
    
    if exist(fullfile(path_session_id,'data.mat'),'file') && exist(fullfile(path_session_id,'behaviorvar.mat'),'file')
        
        mkdir(fullfile(destination,mouse_id,session))
        
        copyfile(fullfile(path_session_id,'data.mat'),fullfile(destination,mouse_id,session))
        copyfile(fullfile(path_session_id,'behaviorvar.mat'),fullfile(destination,mouse_id,session))
        
         if exist(fullfile(path_session_id,'note.txt'),'file')
             copyfile(fullfile(path_session_id,'note.txt'),fullfile(destination,mouse_id,session))
         end
        
    end
        
    
end