% Point-Wise Intensity transformation of all images in the data set
% If you wish to run code with your own images just change the path with the
% folder you wish in your computer.

images = read_images('C:\Users\billi\Desktop\assignment2\MRI_Cardia');
for i = 1:size(images,1)
    img = open_image(images,i);
    
    % Loading pre-calculated probability densities
    load('p_e.mat', 'p_e');
    load('p_G.mat', 'p_G');
    load('p_R.mat', 'p_R');
    
    %1.
    h = image_histogram(img); % Computation of histogram
    p = my_pdf(img, h);       % Computation of Probability Density Function
    q = my_cdf(p);         % Computation of Cumulative Distribution Function
    
    h_m = imhist(img);      % Built-in histogram function
    p_m = my_pdf(img, h_m);    % Calculating PDF of Built-in histogram function
    
    % Calculating Root Mean Squared Error of the PDF's
    RMSE = sqrt(mean((p - p_m).^2));
    display(RMSE);                  % Display the result
    
    
    %2. Estimating pi_0 parameter:
    pi_0 = q(1)/p_e(1);
    
    %3. Computation of CDF for the 2 models
     q1 = gaussian_cdf(pi_0, p_e, p_G);    % Computation of Gaussian CDF
     q2 = rice_cdf(pi_0, p_e, p_R);        % Computation of Rice CDF

     
    %4. Intensity transformation of the 2 models
    T1 = zeros(1,256);
    T2 = zeros(1,256);
    for l = 1:256
        [M1, T1(l)] = min(abs(q1-q(l)));    % Intensity transformation for Gaussian Model
        [M2, T2(l)] = min(abs(q2-q(l)));    % Intensity transformation for Rice Model
    end
    
    %5. Applying Transformation to the image
    Y1 = cast(T1(img + 1) - 1, 'uint8');    % First Model (Gaussian)
    Y2 = cast(T2(img + 1) - 1, 'uint8');    % Second MOdel (Rice)
    
    %6. Histogram Equalization Transform
    %   based on what we discussed in Lecture 10
    [numR, numC] = size(img);    % Dimensions of the image
    pixels = numR * numC;        % Number of all pixels of the image
    cdf_mult = q * 255;          % L = 256 so L-1 = 255
    reference = uint8(cdf_mult); % Rounding the values of new CDF
    
    T = zeros(1,256);
    T = reference;
    
    Y3 = zeros(numR, numC);     % Y3 is going to be the new image
    for j = 1:numR
        for k = 1:numC
            Y3(j,k) = T(img(j,k)+1);    % Applying the T transformation
        end
    end
    Y3 = uint8(Y3);
    
    Y4 = histeq(img);
    
    %7. Creating the figure with the subplots
    figure()
    subplot(2,4,1);
    imshow(img), title("Original image");
    
    subplot(2,4,2);
    imshow(Y1), title("T1 - Gaussian");
    
    subplot(2,4,3);
    imshow(Y2), title("T2 - Rice");
    
    subplot(2,4,4);
    imshow(Y3), title("T - T Transform");
    
%     subplot(2,5,5);
%     imshow(Y4), title("Matlab histeq()");
    
    subplot(2,4,5);
    plot(h), title("Original Histogram");
    
    subplot(2,4,6);
    plot(image_histogram(Y1)), title("T1-Gaussian Histogram");
    
    subplot(2,4,7);
    plot(image_histogram(Y2)), , title("T2-Rice Histogram");
    
    subplot(2,4,8);
    plot(image_histogram(Y3)), title("T-T Transform Histogram");
    

% PLOTS THAT COMPARE MY T IMPLEMENTATION WITH HISTEQ
% Uncomment if you would like to see those plots, I also include 2 of them
% in my report (first 2 pictures in the report)

%     figure();
%     subplot(2,2,1);
%     imshow(Y3), title("T - T Transform");
%     
%     subplot(2,2,2);
%     imshow(Y4), title("Matlab histeq()");
%     subplot(2,2,3);
%     plot(image_histogram(Y3)), title("T-T Transform Histogram");
%     
%     subplot(2,2,4);
%     plot(image_histogram(Y4)), title("histeq Histogram");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some plots that I used in my report, they are commented so if you want
% to uncomment them and see their results in all the images and not just in
% those that I used in my report. If you use them comment the previous so
% there are not so many windows in your screen, thank you!:)


%     figure()
%     subplot(2,5,1);
%     imshow(img), title("Original image");
%     
%     subplot(2,5,2);
%     imshow(Y1), title("T1 - Gaussian");
%     
%     subplot(2,5,3);
%     imshow(Y2), title("T2 - Rice");
%     
%     subplot(2,5,4);
%     imshow(Y3), title("T - T Transform");
    
%     subplot(2,5,5);
%     imshow(Y4), title("Matlab histeq()");
    
%     subplot(2,5,6);
%     plot(h), title("Original Histogram");
%     
%     subplot(2,5,7);
%     plot(image_histogram(Y1)), title("T1-Gaussian Histogram");
%     
%     subplot(2,5,8);
%     plot(image_histogram(Y2)), , title("T2-Rice Histogram");
%     
%     subplot(2,5,9);
%     plot(image_histogram(Y3)), title("T-T Transform Histogram");
    
%     subplot(2,5,10);
%     plot(image_histogram(Y4)), title("histeq Histogram");
%%%%%%%%%%%%%
%

% Matching plot based on the last slide of the Lectures 9-10

%     [~, val] = histeq(img);
%     
%     figure();
%     plot(T);          % my implementation T
%     hold on;
%     plot(val *256);
%     hold on;
%     legend('T', 'histeq()');  % Matlab's implementation
%     title("T, histeq()");

end