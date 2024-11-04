% Function that calculates and returns the mangitude A with different
% values of a, with the formula that our assignment gives
% takes as arguments:
% (1)the alpha values
% (2)the 1st dimension of the image
% (3)the 2nd dimension of the image
function [A] = Alpha(a, M, N)
    % Initializing the output that will returned
    A = zeros(M, N);
    
    % Calculating u, v variables with the given formula
    for m = 1:M
        u = (m - 1)/M;  % Calculating u with the formula that we were given
        for n = 1:N
            v = (n - 1)/N; % Calculating v with the formula that we were given
            % Calcuating the magnitude A with the formula given
            A(m, n) = 1/(1 - a * (cos(2 * pi * u) + cos(2 * pi * v)));
        end
    end
end