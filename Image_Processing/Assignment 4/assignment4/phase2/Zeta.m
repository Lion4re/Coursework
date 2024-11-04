% Function that calculates and returns the linear transformation based on
% the formula of the assignment
% takes as arguments:
% (1)the original image
% (2)the image with applied iDFT
function [Z] = Zeta(X, Y)
    maxX = max(X(:));
    minX = min(X(:));
    maxY = max(Y(:));
    minY = min(Y(:));

    Z = maxX.*(Y-minY)/(maxY-minY) + minX;
end