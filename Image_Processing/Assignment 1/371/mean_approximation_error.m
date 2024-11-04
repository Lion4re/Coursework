function [E] = mean_approximation_error(img, y, S)
    %y == restored image
    
    [numR numC] = size(img);
    
    E = 0;
    for m = 1:numR
        for n = 1:numC
            E = E + (img(m,n) - y(m,n))^2;
        end
    end
    E = sqrt(E);
    E = (1/S) * E;
end