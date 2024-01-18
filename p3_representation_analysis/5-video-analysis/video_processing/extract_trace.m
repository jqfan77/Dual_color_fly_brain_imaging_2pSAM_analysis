function [trace] = extract_trace(video_struct,trace_save_path)
    disp('extract trace!');
    nFrames = size(video_struct,2);
    trace = zeros(1,nFrames);
    %% read one frame every time
    for i = 1:nFrames
        a = video_struct(i).cdata;
        a = single(a);
        trace(i) = mean(a,'all');
    end
    figure();
    plot(trace);
    save(trace_save_path,'trace');
end