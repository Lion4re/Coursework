% Function that applies convolution with box averaging filter (we use a
% 5x5 neighborhood as the assignemnt says) by using my_separable_conv()
% that we implemented earlier and Matlab's fspecial() function.
% takes as arguments:
% (1)the image
% (2)the padding type
function [filtered_img] = my_stddev_filter(img, pad_type)
    % We using the class of 16-bit images in this case, because we have to
    % calculate arguments that will have squared values
    img_16 = uint16(img);
    
    % Creating the Averaging filter with size 5
    twoDimensionalFilter = fspecial('average', 5);
    % Taking the height and width of the filter that we created
    [~, hcol, hrow] = isfilterseparable(twoDimensionalFilter);
    
    % Calculating the arguments J with box averaging (E) and with the help
    % of my_separable_conv with the filter that we created earlier
    first_E = my_separable_conv(img_16.^2, hcol, hrow, pad_type);
    second_E = uint16(my_separable_conv(img_16, hcol, hrow, pad_type)).^2;
    
    % Calculating the J as given in our assignment
    J = double(first_E - second_E);
    J = sqrt(J);
    J = uint8(J);
    
    % Returns the filtered image
    filtered_img = J;
end