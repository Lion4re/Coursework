function [G] = local_difference(img, S)
    [numR numC] = size(img);

    Gh = 0;
    Gv = 0;
    sqrt_diff_ver=0;
    for m = 1:numR
        for n = 2:numC
            sqrt_diff = (img(m,n) - img(m,n-1))^2;
            sqrt_diff_ver = sqrt_diff_ver +(img(n,m)-img(n-1,m))^2;
            Gh = 0 + sqrt_diff;
            
        end
    end
    Gh = sqrt(Gh);
    Gh = (1/S) * Gh;

%     Gv = 0;

%     for m = 2:numR
%         for n = 1:numC
%             sqrt_diff = (img(m,n) - img(m-1,n))^2;
%             Gv = 0 + sqrt_diff;
%         end
%     end

    Gv = sqrt(sqrt_diff_ver);
    Gv = (1/S) * Gv;

    %ypologismos tou g

    G = max(Gh, Gv);
end