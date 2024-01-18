clc;clear;close all
addpath('.');
%%
data_path = 'Y:\0-FJQ\voxel_new\data\5HT-ver16';
filename = 'fly_trace_diff_mean_adjusted.mat';
file_save_name = 'fly_trace_diff_mean_notch_adjusted.mat';
Fs = 30;
Fo = 1;
Q = 35;
%%
folderList=dir(fullfile(data_path));
fileNum=size(folderList,1); 
for k=3:fileNum %% each fly 
	 folder_name = folderList(k).name;
     disp(folder_name);
     load([data_path,'\',folder_name,'\',filename]);
     % notch filter
     [trace_filtered] = notch_filter(trace,Fs,Fo,Q);
     save([data_path,'\',folder_name,'\',file_save_name],'trace_filtered');
end