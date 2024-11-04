% Function that calculates and returns the 1D Discrete Spatial Fourier
% Transform of a given discrete signal s, in the Spatial Domain x, for
% frequencies freq
% takes as arguments:
% (1)the signal
% (2)the spatial domain
% (3)the frequencies

function [DSFT] = mine_1D_DSFT(s, x, freq)
    
    % Define the length of the signal s and
    % the length of the frequencies freq
    signal_sz = size(s, 2);     % Because we have 1D vectors we take the 2nd dimension
    freq_sz = size(freq, 2);
    
    % Initialize the output of the DSFT with respect to the frequencies
    % size (1D)
    DSFT = zeros(1, freq_sz);
    
    % Calculating 1D DSFT based on the slides of the Lectures 17-18
    
    % For each frequency
    for k = 1:freq_sz
        u = freq(k);
        % Loop over the signal and compute the DSFT
        for N = 1:signal_sz
            n = x(N);
            DSFT(k) = DSFT(k) + s(N) * exp(-1j * 2 * pi * u * n);
        end
    end
end