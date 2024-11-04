% Function that calculates the Rice as given in our assignment and
% takes as arguments:
% (1)pi_0
% (2)p_e
% (3)p_R

function [q2] = rice_cdf(pi_0, p_e, p_R)
    p2 = zeros(1,256);
    q2 = zeros(1,256);
    
    
    for i = 1:256
        p2(i) = pi_0 * p_e(i) + (1 - pi_0) * p_R(i);
    end
    
    q2 = my_cdf(p2);

end