% If you wish to run code with your own images just change the path with the
% folder you wish in your computer.
% to panw gia to shmeio me tis eikones


% 1.1 Discrete Space Fourier Transform

% The given 1D filters h(m) and g(m) with their values
h = [0.009 0.027 0.065 0.122 0.177 0.2 0.177 0.122 0.065 0.027 0.009];
g = [0.013 0.028 0.048 0.056 0.039 0 -0.039 -0.056 -0.048 -0.028 -0.013];

% The Discrete Spatial Domain
x = -5:5;

% The Frequency Domain
f = -1/2:0.01:1/2;

% Plotting the two signals (h,g) in the time domain
figure('Name', 'h & g Signals in the time domain');
subplot(1, 2, 1);
plot(x, h), title("h in the time domain");
subplot(1, 2, 2);
plot(x, g), title("g in the time domain");

% Computing the DSFT for h and g signals
h_DSFT = mine_1D_DSFT(h, x, f);
g_DSFT = mine_1D_DSFT(g, x, f);

% In order to plot the magnitude we use the abs function for both signals
H = abs(h_DSFT);
G = abs(g_DSFT);

% Plotting the two signals (H,G) in the frequency domain
figure('Name', 'H & G Signals in the frequency domain');
sgtitle('Using abs() function of h and g to show the magnitude', 'FontSize', 10, 'FontWeight', 'bold');
subplot(1, 2, 1);
plot(f, H), title("H in the frequency domain");
subplot(1, 2, 2);
plot(f, G), title("G in the frequency domain");

% Computing the convolution of the two signals
% First in the time domain with the conv() function
h_t = transpose(h);
conv_hg = conv2(h_t, g);

% Creating the 2D Grid
[X1, X2] = meshgrid(x, x);
figure('Name', 'Convolution of h,g in the time domain');
subplot(1, 1, 1);
surf(X1, X2, conv_hg), title('Convolution of h,g (h*g)');

% Computing the convolution of the two signals
% In the frequency domain calculate convolution as multiplication
h_DSFT_t = transpose(h_DSFT);
conv_hg_DSFT = h_DSFT_t * g_DSFT;

% In order to plot the magnitude we use the abs function
conv_HG_DSFT = abs(conv_hg_DSFT);

% Creating the 2D Grid
[F1, F2] = meshgrid(f, f);
figure('Name', 'Convolution of H,G in the frequency domain');
subplot(1, 1, 1);
surf(F1, F2, conv_HG_DSFT), title('Convolution of H,G (HG)');



% 1.2 Image Segmentation

% If you wish to run code with your own image just change the path with the
% folder you wish in your computer.
images = read_images('images');
img = open_image(images, 1);
image = im2double(img);

% Utilizing the filters h1 & h2
h_transp = transpose(h);
g_transp = transpose(g);
h1 = h_transp * g;
h2 = g_transp * h;

% Computing the filter response y1
y1 = imfilter(image, h_transp);
y1 = imfilter(y1, g);
y1 = im2double(y1);

% Computing the filter response y2
y2 = imfilter(image, g_transp);
y2 = imfilter(y2, h);
y2 = im2double(y2);

% Calculating the sqare magnitude
A = y1.^2 + y2.^2;

% Calculating theta
theta = abs(atan(y2./y1));

% Taking the result of the clustering
L = image_clustering(image, theta, A);

%     Yellow/ Red / Green / Blue
CC = [1 1 0; 1 0 0; 0 1 0; 0 0 1];
% Convert the image (L) into RGB Colored image, 
% with black value equals to black
RGBimg = label2rgb(L, CC, 'black');


% Plotting the figures that we requested to show
figure();
subplot(1, 2, 1);
imshow(image), title('Original Image');
subplot(1, 2, 2);
imshow(RGBimg), title('Cluster Image');