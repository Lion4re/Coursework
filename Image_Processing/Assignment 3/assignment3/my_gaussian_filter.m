% Function that applies a Gaussian filter to an image with the
% my_separable_conv() function that we implemented earlier and with the
% Matlab's function fspecial().
% takes as arguments:
% (1)the image
% (2)the padding type
% (3)the standard deviation parameter σ
function [filtered_img] = my_gaussian_filter(img, pad_type, s)
    
    % Caluclating the size of the filter
    hsize = round(2 * pi * s);
    % Creating the Gaussian filter with size hsize and standard deviation s
    twoDimensionalFilter = fspecial('gaussian', hsize, s);
    
    % Taking the height and width of the filter that we created
    [~, hcol, hrow] = isfilterseparable(twoDimensionalFilter);
    
    % Applying the filter with the help of the my_separable_conv function
    filtered_img = my_separable_conv(img, hcol, hrow, pad_type);
end