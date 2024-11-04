% Function that applies a Gaussian filter to an image by using Matlab's
% imgaussfilt() function.
% takes as arguments:
% (1)the image
% (2)the padding type
% (3)the standard deviation parameter σ
function [filtered_img] = matlab_gaussian_filter(img, pad_type, s)
    % Depend on the argument of padding, we choose the value 0 as an int if
    % the padding argument is zero or if it's replicate then we continue
    % with this padding type with respect to the matlab documentation of
    % function padarray    
    if strcmp(pad_type, 'zero')
        pad_type = 0;
    elseif strcmp(pad_type, 'replicate')
        pad_type = 'replicate';
    end
    
    % Applying the Gaussian Filtering with the arguments that were given
    filtered_img = imgaussfilt(img, s, 'Padding', pad_type);

end