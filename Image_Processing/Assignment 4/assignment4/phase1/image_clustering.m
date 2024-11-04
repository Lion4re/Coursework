% Function that clusters the given image into 5 groups according to the
% conditions of our assignment and returns the clustered image L
% takes as arguments:
% (1)the image
% (2)the unsigned direction of the vector theta
% (3)the square magnitude of filter responses A
function [L] = image_clustering(img, theta, A)

% Taking the size of the image
[d1, d2] = size(img);

% Computing the mean value of A
mi = mean(A(:));

% The initial values that are given to the kmeans
values = [0 0.15*pi 0.35*pi 0.5*pi].';

% Grouping by using the kmeans algorithm
[~, TH] = kmeans(theta(:), 4, 'Distance', 'cityblock', 'Start', values);

% Initialize the output
L = zeros(d1, d2);

% Creating the image n with the conditions of the assignment
for m = 1:d1
    for n = 1:d2
        if A(m, n) <= mi
            L(m, n) = 0;
        elseif theta(m, n) <= 0.5*(TH(1) + TH(2))
            L(m, n) = 1;
        elseif theta(m, n) >= 0.5*(TH(3) + TH(4))
            L(m, n) = 2;
        elseif theta(m, n) > 0.5*(TH(1) + TH(2)) && theta(m, n) <= 0.5*(TH(2) + TH(3))
            L(m, n) = 3;
        elseif theta(m, n) > 0.5*(TH(2) + TH(3)) && theta(m, n) < 0.5*(TH(3) + TH(4))
            L(m, n) = 4;
        end
    end
end

end