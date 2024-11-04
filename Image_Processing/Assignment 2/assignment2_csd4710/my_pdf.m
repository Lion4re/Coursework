% Function that computes the PDF. Takes as arguments: 
% (1)the image
% (2)image's histogram.

function [prob] = my_pdf(img, hs)

    prob = zeros(1,256);

    [numR, numC] = size(img);
    pixels = numR * numC;
    
    for i = 1:256
        prob(i) = hs(i)/pixels;
    end
    
    
    
    
end