% Comparison between Binarized Filtered Images of all images in the data set
% If you wish to run code with your own images just change the path with the
% folder you wish in your computer.

images = read_images('C:\Users\billi\Desktop\assignment3\images');

% Give every image an index for every subquestion of the assignment
img1 = open_image(images, 1);
img2 = open_image(images, 2);
img3 = open_image(images, 3);


% 2.1 Smoothing and Binarization
original_bin = imbinarize(img1);

s = 3;
matlab_bin = imbinarize(matlab_gaussian_filter(img1, 'replicate', s));
my_bin = imbinarize(my_gaussian_filter(img1, 'replicate', s));

% Creating the figure with the subplots
figure();
sgt = sgtitle('2.1 Smoothing and Binarization');

subplot(3, 3, 1);
imshow(original_bin), title("Original Binarized Image");
subplot(3, 3, 2);
imshow(matlab_bin), title("Matlab Gaussian, σ = 3");
subplot(3, 3, 3);
imshow(my_bin), title("My Gaussian, σ = 3");

s = 5;
matlab_bin = imbinarize(matlab_gaussian_filter(img1, 'replicate', s));
my_bin = imbinarize(my_gaussian_filter(img1, 'replicate', s));

subplot(3, 3, 4);
imshow(original_bin), title("Original Binarized Image");
subplot(3, 3, 5);
imshow(matlab_bin), title("Matlab Gaussian, σ = 5");
subplot(3, 3, 6);
imshow(my_bin), title("My Gaussian, σ = 5");

s = 7;
matlab_bin = imbinarize(matlab_gaussian_filter(img1, 'replicate', s));
my_bin = imbinarize(my_gaussian_filter(img1, 'replicate', s));

subplot(3, 3, 7);
imshow(original_bin), title("Original Binarized Image");
subplot(3, 3, 8);
imshow(matlab_bin), title("Matlab Gaussian, σ =7");
subplot(3, 3, 9);
imshow(my_bin), title("My Gaussian, σ = 7");

% ===========================================
% 2.2 Local Standard Deviation and Binarization
original_bin = imbinarize(img2);
matlab_bin = imbinarize(matlab_stddev_filter(img2, 'replicate'));
my_bin = imbinarize(my_stddev_filter(img2, 'replicate'));

% Creating the figure with the subplots
figure();
sgt = sgtitle('2.2 Local Standard Deviation and Binarization');
subplot(1, 3, 1);
imshow(original_bin), title("Original Binarized Image");
subplot(1, 3, 2);
imshow(matlab_bin), title("Matlab Standard Deviation");
subplot(1, 3, 3);
imshow(my_bin), title("My Standard Deviation");

% ===========================================
% 2.3 Laplacian and Binarization

original_bin = imbinarize(img3);

s2 = 1;
matlab_bin = imbinarize(matlab_log_filter(img3, 'replicate', s2));
my_bin = imbinarize(my_log_filter(img3, 'replicate', s2));

% Creating the figure with the subplots
figure();
sgt = sgtitle('2.3 Laplacian and Binarization');

subplot(4, 3, 1);
imshow(original_bin), title("Original Binarized Image");
subplot(4, 3, 2);
imshow(matlab_bin), title("Matlab Laplacian, σ2 = 1");
subplot(4, 3, 3);
imshow(my_bin), title("My Laplacian, σ2 = 1");

s2 = sqrt(2);
matlab_bin = imbinarize(matlab_log_filter(img3, 'replicate', s2));
my_bin = imbinarize(my_log_filter(img3, 'replicate', s2));

subplot(4, 3, 4);
imshow(original_bin), title("Original Binarized Image");
subplot(4, 3, 5);
imshow(matlab_bin), title("Matlab Laplacian, σ2 = " + s2);
subplot(4, 3, 6);
imshow(my_bin), title("My Laplacian, σ2 = " + s2);

s2 = 2;
matlab_bin = imbinarize(matlab_log_filter(img3, 'replicate', s2));
my_bin = imbinarize(my_log_filter(img3, 'replicate', s2));

subplot(4, 3, 7);
imshow(original_bin), title("Original Binarized Image");
subplot(4, 3, 8);
imshow(matlab_bin), title("Matlab Laplacian, σ2 = 2");
subplot(4, 3, 9);
imshow(my_bin), title("My Laplacian, σ2 = 2");

s2 = 2 * sqrt(2);
matlab_bin = imbinarize(matlab_log_filter(img3, 'replicate', s2));
my_bin = imbinarize(my_log_filter(img3, 'replicate', s2));

subplot(4, 3, 10);
imshow(original_bin), title("Original Binarized Image");
subplot(4, 3, 11);
imshow(matlab_bin), title("Matlab Laplacian, σ2 = " + s2);
subplot(4, 3, 12);
imshow(my_bin), title("My Laplacian, σ2 = " + s2);

