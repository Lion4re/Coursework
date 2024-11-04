% Function that computes the histogram of an image and takes as argument:
% (1)the image

function [hs] = image_histogram(img)

    hs = zeros(1,256);

    [numR, numC] = size(img);

    for i = 1:256
        for j = 1:numR
            for k = 1:numC
                if (i-1) == img(j,k)
                    hs(i) = hs(i)+1;
                end
            end
        end
    end
end