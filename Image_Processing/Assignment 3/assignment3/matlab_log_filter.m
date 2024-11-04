% Function that applies a Laplacian of Gaussian Filter to an image with the
% help of Matlab's fspecial() and imfilter() functions.
% takes as arguments:
% (1)the image
% (2)the padding type
% (3)the standard deviation parameter s2
function [filtered_img] = matlab_log_filter(img, pad_type, s2);

    % Depend on the argument of padding, we choose the value 0 as an int if
    % the padding argument is zero or if it's replicate then we continue
    % with this padding type with respect to the matlab documentation of
    % function padarray
    if strcmp(pad_type, 'zero')
        pad_type = 0;
    elseif strcmp(pad_type, 'replicate')
        pad_type = 'replicate';
    end
   
    % Calculating s1, which is equal as s2 times 1.28 as given in the
    % assignment
    s1 = s2 * 1.28;
    
    % Caluclating the size of the first (G1) filter
    hsizeG1 = round(2 * pi * s1);
    % Creating the Gaussian filter (G1) with size hsizeG1 and 
    % standard deviation s1
    twoDimensionalFilterG1 = fspecial('gaussian', hsizeG1, s1);
    
    % Caluclating the size of the second (G2) filter
    hsizeG2 = round(2 * pi * s2);
    % Creating the Gaussian filter (G2) with size hsizeG2 and 
    % standard deviation s2
    twoDimensionalFilterG2 = fspecial('gaussian', hsizeG2, s2);
    
    % Applying the filters to the image seperatly
    G1 = imfilter(img, twoDimensionalFilterG1, pad_type);
    G2 = imfilter(img, twoDimensionalFilterG2, pad_type);
    % Calculate the Laplacian as given from the assignment (difference of
    % G1, G2)
    filtered_img = G1 - G2;
end