function [data_filtered] = notch_filter(x,Fs,Fo,Q)
    Wo = Fo/(Fs/2);
    BW = Wo/Q;
    [b, a] = iircomb(Fs/Fo, BW, 'notch');
    fvtool(b,a);
    data_filtered = filter(b, a, x);
    figure();plot(x);
    figure();plot(data_filtered);
    close all
end