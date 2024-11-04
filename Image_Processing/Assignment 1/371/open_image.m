function [img] = open_image(images,idx)
    img = cell2mat(images(idx));
    img = mat2gray(img);
end