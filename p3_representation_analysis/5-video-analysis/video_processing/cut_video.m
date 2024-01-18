function [mov,Rate] = cut_video(path_in,path_out,H,W,UP,LEFT)
    if exist(path_out,'file')
        disp('video exist!');
        video = VideoReader(path_out);%%%
        nFrames = video.NumFrames;   %得到帧数
        Rate = video.FrameRate;
        % Preallocate movie structure.
        mov(1:nFrames) = struct('cdata',zeros(video.Height,video.Width,3,'uint8'),'colormap',[]);
        %% read one frame every time
        for i = 1:nFrames
            a = read(video,i);
            mov(i).cdata = a;
    %         P = mov(i).cdata;
        %     imshow(P),title('视频');
        end
    else
        disp('cut video!');
        video = VideoReader(path_in);%%%
        nFrames = video.NumFrames;   %得到帧数
        Rate = video.FrameRate;
        % Preallocate movie structure.
        mov(1:nFrames) = struct('cdata',zeros(H,W,3,'uint8'),'colormap',[]);
        %% read one frame every time
        for i = 1:nFrames
            a = read(video,i);
            aa = a(UP:UP+H,LEFT:LEFT+W,:);
            mov(i).cdata = aa;
    %         P = mov(i).cdata;
        %     imshow(P),title('视频');
        end
        %% 存一下数据light_on_frame到AVI  
        writerObj =VideoWriter(path_out); % 生成一个avi动画
        writerObj.FrameRate=Rate; % 设置avi动画的参数，设置帧速率
        open(writerObj); % 打开avi动画
        writeVideo(writerObj,mov); % 将保存的动画写入到视频文件中
        close(writerObj); % 关闭动画
    end
end