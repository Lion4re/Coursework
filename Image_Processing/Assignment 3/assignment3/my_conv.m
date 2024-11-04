% Function that returns the Valid Convolution of an image with the 2D
% filter kernel
% takes as arguments:
% (1)the image
% (2)the kernel

function [conv_img] = my_conv(img, kernel)
    
    % Taking the height and width of the kernel
    [k_r, k_c] = size(kernel);
    
    % Converting image to double precision
    img = double(img);
    % Rearreange image blocks into columns with respect to our kernel
    img_col = im2col(img, [k_r, k_c], 'sliding');
    
    % Flipping the kernel left-right and up-down
    kernel_lr = fliplr(kernel);
    kernel_lrud = flipud(kernel_lr);
    
    % Taking the height and width of the image
    [img_r, img_c] = size(img);
    
    % Calculating the output height and width with the right dimensions of
    % valid convolution (as in lecture)
    conv_r = img_r - k_r + 1;
    conv_c = img_c - k_c + 1;
    
    % Creating the output image
    conv_img = zeros(conv_r, conv_c);
    
    % Calculating the valid convolution between the given image and kernel
    for i = 1:conv_c
        for j = 1:conv_r
            f = img_col(:, j + (i - 1) * conv_r);
            g = kernel_lrud(:);
            conv_img(j, i) = dot(f, g);
        end
    end
      
end