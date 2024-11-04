% Function that returns the Same-Size Convolution of an image with the 2D
% filter kernel, which is a separable filter, with 1D horizontal and 1D
% vertical filters.
% takes as arguments:
% (1)the image
% (2)the 1D horizontal kernel
% (3)the 1D vertical kernel
% (4)the padding type
function [conv_img] = my_separable_conv(img, hor_kernel, ver_kernel, pad_type)
    
    % Taking the dimensions of each 1D kernel
    k_r = size(hor_kernel, 1);
    k_c = size(ver_kernel, 2);
    
    % Calculating the output height and width with the right dimensions of
    % Same-Size Convolution (as in lecture)
    conv_r = round((k_r - 1) / 2);
    conv_c = round((k_c - 1) / 2);
    
    % Depend on the argument of padding, we choose the value 0 as an int if
    % the padding argument is zero or if it's replicate then we continue
    % with this padding type with respect to the matlab documentation of
    % function padarray
    if strcmp(pad_type, 'zero')
        pad_type = 0;
    elseif strcmp(pad_type, 'replicate')
        pad_type = 'replicate';
    end
    
    % Appying padding
    conv_img = padarray(img, [conv_r, conv_c], pad_type, 'both');
    
    % Calculating the convolution with the 1D kernels
    conv_ver = my_conv(conv_img, ver_kernel);
    conv_verhor = my_conv(conv_ver, hor_kernel);
    % conv_img is our output so give the final value of convolutions there
    conv_img = conv_verhor;
    
    % "reshape" our convolved image as a one-dimensional column vector with 
    % (:) operator and
    % check the max value compared to L=256 so we can choose the class of
    % our image, between 8-bit and 16-bit class
    if max(conv_img(:)) < 256
        conv_img = uint8(conv_img);
    elseif max(conv_img(:)) >= 256
        conv_img = uint16(conv_img);
    end
    
end