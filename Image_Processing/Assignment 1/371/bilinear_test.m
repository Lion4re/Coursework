%If you wish to run code with your own images just change the path with the
%folder you wish in your computer.
images = read_images('C:\Users\billi\Desktop\371\Brodatz');
for i = 1:size(images,1)
    img = open_image(images,i);
    
    %img == original image
    %restored == restored by my functions image
    %matlab == matlab imresize image
    
    %Subsampling with my nearest neighbor function and upsampling with my
    %bilinear function
    sub_image = my_imresize_NN(img, 8, "sub");
    restored = my_imresize_Bilinear(sub_image, 8, "up");
    %Subsampling and upsampling with imresize function
    matlab = imresize(img, 1/8);
    matlab = imresize(matlab, 8);
    
    %Calculating the metrics for my function image restoration and imresize
    %function
    S = S_calculator(restored);
    E = mean_approximation_error(restored, matlab, S);
    G = local_difference(restored, S);

    mS = S_calculator(matlab);
    mG = local_difference(matlab, mS);
    
    %Printing the metrics
    disp(num2str(i)+".Error= "+num2str(E)+". My G="+ num2str(G)+". Matlab G="+num2str(mG));
    
    % Tested with some images, uncomment if you want to see the images
%     figure();
%     imshow(restored), title("bilinear Restored Image");
%     figure();
%     imshow(matlab), title("Matlab's imresize Restored Image");
%     figure();
%     imshow(img), title("Original Image");

end