% Function to uniformly quantize the phase of an image and gives as output
% the quantized image
% takes as arguments:
% (1)the image to be quantized (grayscale)
% (2)number of levels to use for quantization
function image_q = phase_u_quantize(image, levels)
    % Computing the Fourier transform of the image
    F = fft2(image);
    
    % Computing the range of the phase
    min_angle = min(min(angle(F)));
    max_angle = max(max(angle(F)));
    range = max_angle - min_angle;
    
    % Quantizing the phase
    angle_q = round((angle(F) - min_angle) * levels / range) * range / levels + min_angle;
    
    % Replacing the phase of the Fourier transform with the quantized phase
    F_q = abs(F) .* exp(1i * angle_q);
    
    % Computing the inverse Fourier transform of the quantized Fourier transform
    image_q = abs(ifft2(F_q));
end