function [trace,motion] = extract_trace_diff(video_struct,frame_start,frame_stop,Rate,output_frame_start,output_frame_stop,...
                                   trace_save_path,motion_save_path,avi_save_path)
    if frame_stop == inf
        frame_stop = size(video_struct,2);
    end
    nFrames = frame_stop-frame_start+1;
    if output_frame_stop == inf
        output_frame_stop = nFrames;
    end
    a = video_struct(1).cdata; 
    Hight = size(a,1);
    Width = size(a,2);
    %%
    trace = zeros(1,nFrames);
    motion = zeros(nFrames,Hight,Width,'single');
    %% read one frame every time
    a0 = 0;
    for i = frame_start:frame_stop
        a = video_struct(i).cdata;
%         a = rgb2gray(a);
        a = single(a);
%     %     disp(size(a));
        a = mean(a,3);
        flag = abs(a-a0);
        trace(i-frame_start+1) = mean(flag,'all');
        motion(i-frame_start+1,:,:) = flag;
        a0 = a;
    end
    trace = trace(2:end);
    motion = motion(2:end,:,:,:);
    figure();
    plot(trace);
    print(gcf, '-dpng', '-r600', [trace_save_path,'.png'])  
    save(trace_save_path,'trace');
    save(motion_save_path,'motion');
    %%
    nFrames_avi = output_frame_stop-output_frame_start+1;
    % Preallocate movie structure.
    mov(1:nFrames_avi) = struct('cdata',zeros(Hight,Width,'uint8'),'colormap',jet());
    % read one frame every time
    for i = 1:nFrames_avi
        mov(i).cdata = uint8(squeeze(motion(i,:,:)));
    end
    % 存一下数据light_on_frame到AVI  
    writerObj =VideoWriter(avi_save_path); % 生成一个avi动画
    writerObj.FrameRate=Rate; % 设置avi动画的参数，设置帧速率
    % writerObj.Colormap = summer(256);
    open(writerObj); % 打开avi动画
    writeVideo(writerObj,mov); % 将保存的动画写入到视频文件中
    close(writerObj); % 关闭动画
end