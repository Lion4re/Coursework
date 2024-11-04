function [S] = S_calculator(img)
    [numR numC] = size(img);
    
    S = 0;
    for m = 1:numR
        for n = 1:numC
            S = S + img(m,n)^2;
        end
    end
    S = sqrt(S);
end