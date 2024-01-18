clc; clear; close all
addpath('.');
%% 读入含有led信号的视频
data_path = 'Y:\0-FJQ\voxel_new\data\5HT-ver16\20230513-r5HT1.0-fly1';
fly_video_name = 'fly.avi';
light_trace_name = 'light_trace_thresh.mat';
light_trace_name_raw = 'light_trace.mat';
fly_trace_name_new = 'fly_trace_adjusted.mat';
fly_mean_motion_name = 'fly_trace_diff_mean_adjusted.mat';
fly_motion_name = 'fly_trace_diff_adjusted.mat';
fly_motion_avi_name = 'fly_trace_diff_adjusted.avi';
% 
frame_start_motion = 1;
frame_stop_motion = inf;
output_frame_start = 1;
output_frame_stop = 500;
rate = 30;
%% read video
video = VideoReader([data_path,' \',fly_video_name]);%%%
nFrames = video.NumFrames;   %得到帧数
Rate = video.FrameRate;
% Preallocate movie structure.
mov(1:nFrames) = struct('cdata',zeros(video.Height,video.Width,3,'uint8'),'colormap',[]);
for i = 1:nFrames
    a = read(video,i);
    mov(i).cdata = a;
end
%% load trace and adjust 
load([data_path,'\',light_trace_name]);
fly_mean_trace = extract_trace(mov,fly_trace_name_new);
high_level = mean(fly_mean_trace(trace_processed==1));
low_level = mean(fly_mean_trace(trace_processed==0));
load([data_path,'\',light_trace_name_raw]);
trace = trace-mean(trace(trace_processed==0));
trace = trace/(mean(trace(trace_processed==1)))*(high_level-low_level);
%% adjust video - substract
mov_new = mov;
for i = 1:nFrames
    a = single(mov(i).cdata);
    a = a-trace(i);
    mov_new(i).cdata = a;
end
fly_trace = extract_trace(mov_new,[data_path,'/',fly_trace_name_new]);
%% adjust video - strange value
thresh_high = 156;
thresh_low = 153.25;
inds = find(fly_trace>thresh_high | fly_trace<thresh_low);
% for i = 102:length(inds)
for i = 45:912
    mov_new(inds(i)).cdata = mov_new(inds(i)).cdata-(fly_trace(inds(i))-fly_trace(inds(i)-1));
    fly_trace(inds(i)) = fly_trace(inds(i)-1);
end
fly_trace = extract_trace(mov_new,[data_path,'/',fly_trace_name_new]);
%% extract fly motion
[mean_motion,motion] = extract_trace_diff(mov_new,frame_start_motion,frame_stop_motion,rate,output_frame_start,output_frame_stop,...
                                   [data_path,'/',fly_mean_motion_name],...
                                   [data_path,'/',fly_motion_name],...
                                   [data_path,'/',fly_motion_avi_name]);