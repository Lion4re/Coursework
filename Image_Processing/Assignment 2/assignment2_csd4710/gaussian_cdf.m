% Function that calculates the Gaussian as given in our assignment and
% takes as arguments:
% (1)pi_0
% (2)p_e
% (3)p_G

function [q1] = gaussian_cdf(pi_0, p_e, p_G)
    p1 = zeros(1,256);
    q1 = zeros(1,256);
    
    
    for i = 1:256
        p1(i) = pi_0 * p_e(i) + (1 - pi_0) * p_G(i);
    end
    
    q1 = my_cdf(p1);

end