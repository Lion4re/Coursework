function [resized_image] = my_imresize_Bilinear(I, a, op)
    %scaling the image values(intensities) in the range [0,1]
    img = im2double(I);
    
    %if op is equal to up then upsample by the factor a using bilinear
    %interpolation, if op is equal to sub then subsample by nearest
    %neighbor
    if op == "up"
        %Taking the size of the original image
        %numR == number of rows
        %numC == number of columns
        [numR numC] = size(img);

        %Determine the new image dimensions based on the factor a
        new_x = numR * a;
        new_y = numC * a;

        %Finding the ratio of the new size
        ratioR = new_x./(numR - 1);
        ratioC = new_y./(numC -1);

        %Make a blank output
        out = zeros(new_x, new_y);

        %Generating the output
        %Calculating the weight of the points based on the bilinear
        %interpolation.
        %W stands for the width distance of the pixel
        %H stands for the height distance of the pixel
        for countR = 0:new_x-1
            for countC = 0:new_y-1
                 W = -(((countR./ratioR)-floor(countR./ratioR))-1);
                 H = -(((countC./ratioC)-floor(countC./ratioC))-1);
                 I11 = img(1+floor(countR./ratioR),1+floor(countC./ratioC));
                 I12 = img(1+ceil(countR./ratioR),1+floor(countC./ratioC));
                 I21 = img(1+floor(countR./ratioR),1+ceil(countC./ratioC));
                 I22 = img(1+ceil(countR./ratioR),1+ceil(countC./ratioC));
                 out(countR+1,countC+1) = (1-W).*(1-H).*I22 + (W).*(1-H).*I21 + (1-W).*(H).*I12 + (W).*(H).*I11;
            end
        end
        %The final image that will get returned
        resized_image = out;
    elseif op == "sub"
        a = 1/a;
        %Taking the size of the original image
        %numR == number of rows
        %numC == number of columns
        [numR numC] = size(img);

        %Determine the new image dimensions based on the factor a
        new_x = numR * a;
        new_y = numC * a;

        %Finding the ratio of the new size
        ratioR = new_x./(numR - 1);
        ratioC = new_y./(numC - 1);

        %Make a blank output
        out = zeros(new_x, new_y);

        %Generating the output
        for countR = 0:new_x-1
            for countC = 0:new_y-1
                out(countR+1, countC+1) = img(1+round(countR./ratioR), 1+round(countC./ratioC));
            end
        end

        %The final image that will get returned
        resized_image = out;
    end  
end