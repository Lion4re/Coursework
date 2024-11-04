% Function that applies convolution with box averaging filter (we use a
% 5x5 neighborhood as the assignemnt says) by using Matlab's fspecial and
% imfilter() functions.
% takes as arguments:
% (1)the image
% (2)the padding type
function [filtered_img] = matlab_stddev_filter(img, pad_type)
    % We using the class of 16-bit images in this case, because we have to
    % calculate arguments that will have squared values
    img_16 = uint16(img);

    % Depend on the argument of padding, we choose the value 0 as an int if
    % the padding argument is zero or if it's replicate then we continue
    % with this padding type with respect to the matlab documentation of
    % function padarray
    if strcmp(pad_type, 'zero')
        pad_type = 0;
    elseif strcmp(pad_type, 'replicate')
        pad_type = 'replicate';
    end
    
    % Creating the Averaging filter with size 5
    twoDimensionalFilter = fspecial('average', 5);
    
    % Calculating the arguments J with box averaging (E) and with the help
    % of Matlab's imfilter() function with the filter that we created
    % earlier
    first_E = imfilter(img_16.^2, twoDimensionalFilter, pad_type);
    second_E = imfilter(img_16, twoDimensionalFilter, pad_type).^2;
    
    % Calculating the J as given in our assignment
    J = double(first_E - second_E);
    J = sqrt(J);
    J = uint8(J);
    
    % Returns the filtered image
    filtered_img = J;
end