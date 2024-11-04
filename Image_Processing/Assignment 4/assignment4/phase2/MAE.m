% Function that computes and returns the Mean Absolute Error (MAE) between
% the resulting image (Y) and the original one (1)
% takes as arguments:
% (1)the original image X
% (2)the resulting image Y
function [error] = MAE(X, Y)
    % Taking the dimensions of the image
    [M, N] = size(X);
    
    % Calculating the MAE
    error = 0;
    for i = 1:M
        for j = 1:N
            error = error + abs(X(i,j) - Y(i,j));
        end
    end
    error = error/(M * N);
end
