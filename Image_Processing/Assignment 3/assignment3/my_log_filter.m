% Function that applies a Laplacian of Gaussian Filter to an image with the
% help of my_separable_conv() function that we implemented earlier
% and the Matlab's fspecial() function.
% takes as arguments:
% (1)the image
% (2)the padding type
% (3)the standard deviation parameter s2
function [filtered_img] = my_log_filter(img, pad_type, s2)
    % Calculating s1, which is equal as s2 times 1.28 as given in the
    % assignment
    s1 = s2 * 1.28;
    
    % Caluclating the size of the first (G1) filter
    hsizeG1 = 2 * floor((2 * pi * s1) / 2) + 1;
    % Creating the Gaussian filter (G1) with size hsizeG1 and 
    % standard deviation s1
    twoDimensionalFilterG1 = fspecial('gaussian', hsizeG1, s1);
    
    % Taking the height and width of the filter G1 that we created
    [~, hcolG1, hrowG1] = isfilterseparable(twoDimensionalFilterG1);
    
    % Caluclating the size of the second (G2) filter
    hsizeG2 = 2 * floor((2 * pi * s2) / 2) + 1;
    % Creating the Gaussian filter (G2) with size hsizeG2 and 
    % standard deviation s2
    twoDimensionalFilterG2 = fspecial('gaussian', hsizeG2, s2);
    
    % Taking the height and width of the filter G2 that we created
    [~, hcolG2, hrowG2] = isfilterseparable(twoDimensionalFilterG2);
    
    % Applying the filters to the image seperatly
    G1 = my_separable_conv(img, hcolG1, hrowG1, pad_type);
    G2 = my_separable_conv(img, hcolG2, hrowG2, pad_type);
    % Calculate the Laplacian as given from the assignment (difference of
    % G1, G2)
    filtered_img = G1 - G2;
end