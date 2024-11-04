% Function that computes the PDF. Takes as argument: 
% (1)the PDF

function [q] = my_cdf(p)
    q = zeros(1,256);
    
    q(1) = p(1);
    sum = q(1);
    for i = 2: 256
        sum = sum + p(i);
        q(i) = sum;
    end
end