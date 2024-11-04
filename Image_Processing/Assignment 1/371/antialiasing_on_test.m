images = read_images('C:\Users\billi\Desktop\371\Brodatz');
for i = 1:size(images,1)
    img = open_image(images,i);
    %Subsampling and upsampling with Antialiasing on
    sub_image = imresize(img, 1/8, 'nearest', 'Antialiasing', true);
    restored = imresize(sub_image, 8, 'nearest');
    
    %Subsampling and upsampling with Antialiasing off
    antialiasing_off_sub = imresize(img, 1/8, 'nearest', 'Antialiasing', false);
    antialiasing_off = imresize(antialiasing_off_sub, 8, 'nearest');
   
    %Calculating the metrcis for Antialiasing on and Original image
    S = S_calculator(restored);
    E = mean_approximation_error(restored, img , S);
    G = local_difference(sub_image, S);
    
    %Calculating the metrics for Antialiasing off and Original image
    S2 = S_calculator(antialiasing_off);
    E2 = mean_approximation_error(antialiasing_off, img, S);
    
    %Calculating original image local difference
    oS = S_calculator(img);
    oG = local_difference(img, oS);
    
    %Printing the metrics
    disp(num2str(i)+".Error of Antialiasing On with Original Image= "+num2str(E));
    disp("Error of Antialiasing Off with Original Image=" + num2str(E2));
    disp("Original image G="+num2str(oG));
    disp("--------------------------------------------");
    
    % Tested with some images, uncomment if you want to see the images
%     figure();
%     imshow(restored), title("AA ON");
%     figure();
%     imshow(antialiasing_off), title("AA OFF");
%     figure();
%     imshow(img), title("Original Image");

end