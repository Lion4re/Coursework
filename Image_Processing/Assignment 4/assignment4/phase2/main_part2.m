% If you wish to run code with your own images just change the path with the
% folder you wish in your computer.

images = read_images('images');

% WARNING: The warning that pops up is only on the einstein image, all
% other images do not give any warning. But still the results are those
% that we want to have. It's not a problem at all and it's because of the
% Fourier Transform.

for img_idx = 1:numel(images)
    img = open_image(images, img_idx);
    img = im2double(img);

    % Taking the dimensions of the image
    [M, N] = size(img);
    % Computing the 2-D DFT of the image
    F = fft2(img);

    % 2.1 The role of the phase and magnitude
    % Part 1

    % Set of values a
    a = [0.45 0.49 0.495];

    % Creating 2 cells to store the values for all cases of a
    A = cell(3, 1); % for the magnitude
    Z = cell(3, 1); % for the transformed images

    % Creating an array to store the MAErrors
    errors = zeros(3, 1);
    for i = 1:numel(a)
        A{i} = Alpha(a(i), M, N);

        % Applying iDFT for the magnitude case i
        Y = ifft2(A{i} .* exp(1j.*angle(F)), 'symmetric');

        % Calculating the resulting images for each case
        Z{i} = Zeta(img, Y);
        % Adjusting pixel values to achieve histogram matching
        Z{i} = histeq(Z{i}, imhist(img));
        % Computing MAE between the resulting and original image
        errors(i) = MAE(img, Z{i});
    end

    % Creating the figure with the orignal image and the resulting images for
    % each value a
    figure();
    subplot(1, 4, 1);
    imshow(img), title('Original Image');
    subplot(1, 4, 2);
    imshow(Z{1}), title("Image with a=" + a(1) + " and Mean Abs. Error=" + errors(1));
    subplot(1, 4, 3);
    imshow(Z{2}), title("Image with a=" + a(2) + " and Mean Abs. Error=" + errors(2));
    subplot(1, 4, 4);
    imshow(Z{3}), title("Image with a=" + a(3) + " and Mean Abs. Error=" + errors(3));


    % Part 2

    % Intervals set K_ph
    K_ph = [5 9 17 33 65];

    % Creating a cell to store the images:
    % legend :  (i, 1) == magnitude A with a = 0.495
    %           (i, 2) == magnitude of the original image
    quantized = cell(5, 2);

    % Creating an array to store the MAE between all the transformed images and
    % the original image
    errors = zeros(5, 2);

    % Uniform quantization of the phase for the set K with the help of function
    % phase_u_quantize that I implemented.
    % Also computes the MAE between all transformed images and the original one
    for i = 1:numel(K_ph)
        quantized{i, 1} = phase_u_quantize(Z{3}, K_ph(i));
        quantized{i, 2} = phase_u_quantize(img, K_ph(i));

        errors(i, 1) = MAE(img, quantized{i, 1});
        errors(i, 2) = MAE(img, quantized{i, 2});
    end

    % Creating the figure with the resulting images for each value K_ph, and
    % the MAE between all transformed images and the original one
    figure();
    subplot(2, 5, 1);
    imshow(quantized{1, 1});
    title({"a = 0.495,", "phase quantization K_Φ = 5, E = " + errors(1, 1)});
    subplot(2, 5, 2);
    imshow(quantized{2, 1});
    title({"a = 0.495,", "phase quantization K_Φ = 9, E = " + errors(2, 1)});
    subplot(2, 5, 3);
    imshow(quantized{3, 1});
    title({"a = 0.495,", "phase quantization K_Φ = 17, E = " + errors(3, 1)});
    subplot(2, 5, 4);
    imshow(quantized{4, 1});
    title({"a = 0.495,", "phase quantization K_Φ = 33, E = " + errors(4, 1)});
    subplot(2, 5, 5);
    imshow(quantized{5, 1}), title({"a = 0.495,", "phase quantization K_Φ = 65, E = " + errors(5, 1)});

    subplot(2, 5, 6);
    imshow(quantized{1, 2});
    title({"Phase quantization K_Φ = 5, E = " + errors(1, 2)});
    subplot(2, 5, 7);
    imshow(quantized{2, 2});
    title({"Phase quantization K_Φ = 9, E = " + errors(2, 2)});
    subplot(2, 5, 8);
    imshow(quantized{3, 2});
    title({"Phase quantization K_Φ = 17, E = " + errors(3, 2)});
    subplot(2, 5, 9);
    imshow(quantized{4, 2});
    title({"Phase quantization K_Φ = 33, E = " + errors(4, 2)});
    subplot(2, 5, 10);
    imshow(quantized{5, 2});
    title("Quantized with K_Φ = 65, E = " + errors(5, 2));

    % 2.2 Image Compression

    p = [2.5 5 7.5];

    % Creating an array to store the MAErrors
    errors = zeros(3, 1);

    % Creating a cell to store the reconstructed images
    reconstructed_image = cell(3, 1);

    % Loop over the values of p
    for i = 1:numel(p)
        % Calculate the magnitude
        magnitude = abs(F);
        % Sorting the magnitude in decreasing order
        [~, indices] = sort(magnitude(:), 'descend');

        % Select the top p% of the spatial frequencies with the largest magnitude
        top_p = magnitude > magnitude(indices(round(p(i) * length(indices) / 100)));

        % Set the remaining spatial frequencies to zero
        magnitude(~top_p) = 0;

        % Reconstructing the DFT coefficients using the modified magnitude
        dft_coeffs_modified = magnitude .* exp(1i * angle(F));

        % Computing the inverse DFT of the modified DFT coefficients to obtain the reconstructed image
        reconstructed_image{i} = ifft2(dft_coeffs_modified);
        % Calculating the MAE between original and reconstructed images
        errors(i) = MAE(img, reconstructed_image{i});
    end

    % Creating the figure with the compressed images for each value p, and
    % the MAE between all compressed images and the original one
    figure;
    subplot(1, 4, 1);
    imshow(img);
    title('Original Image');
    subplot(1, 4, 2);
    imshow(reconstructed_image{1});
    title("p = 2.5, E = " + errors(1));
    subplot(1, 4, 3);
    imshow(reconstructed_image{2});
    title("p = 5, E = " + errors(2)); 
    subplot(1, 4, 4);
    imshow(reconstructed_image{3});
    title("p = 7.5, E = " + errors(3));
    
    % Plotting magnitude of the image (actually the log of magnitude)
    F_shifted = fftshift(F);
    magnitude = abs(F_shifted);
    log_magnitude = log(magnitude);
    figure;
    imshow(log_magnitude), title('Logarithm of Magnitude');
end
