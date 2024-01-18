clc;clear;close all
%%
data_path = 'Y:\0-FJQ\voxel_new\data\5HT-ver16';
filename = 'light_trace.mat';
file_save_name = 'light_trace_thresh.mat';
%%
folderList=dir(fullfile(data_path));
fileNum=size(folderList,1); 
for k=3+2:fileNum %% each fly 
	 folder_name = folderList(k).name;
     disp(folder_name);
     load([data_path,'\',folder_name,'\',filename]);
     % process
     thresh = (max(trace)+min(trace))/2;
%      thresh = 95.5;
     trace_processed = trace>thresh;
     figure();plot(trace);
     figure();plot(trace_processed);
     % manual
     save([data_path,'\',folder_name,'\',file_save_name],'trace_processed');
     close all
end