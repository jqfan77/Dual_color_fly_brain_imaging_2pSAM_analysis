clc;clear;close all
addpath('.')
%% 读入含有led信号的视频
data_path = 'Y:\0-FJQ\voxel_new\data\5HT-ver16\20230603-r5HT1.0-fly1';
video_name = 'WIN_20230603_14_09_58_Pro.mp4';
fly_video_name = 'fly.avi';
light_video_name = 'light.avi';
fly_trace_name = 'fly_trace.mat';
light_trace_name = 'light_trace.mat';
fly_mean_motion_name = 'fly_trace_diff_mean.mat';
fly_motion_name = 'fly_trace_diff.mat';
fly_motion_avi_name = 'fly_trace_diff.avi';
% fly
H_fly = 100;
W_fly = 100;
UP_fly = 215;
LEFT_fly = 595;
% light
H_light = 200;
W_light = 200;
UP_light = 1;
LEFT_light = 1070;
%% cut video
cut_video([data_path,'/',video_name],[data_path,'/',fly_video_name],H_fly,W_fly,UP_fly,LEFT_fly);