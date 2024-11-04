function [resized_image] = my_imresize_NN(I, a, op)
    %scaling the image values(intensities) in the range [0,1]
    img = im2double(I);
    
    %if op value is sub then subsample by the factor of 1/a
    if op == "sub"
        a = 1/a;
    end
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
