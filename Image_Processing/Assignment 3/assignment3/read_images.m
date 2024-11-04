function [imgs] = read_images(folder)
    imds = imageDatastore(folder,"FileExtensions",".png");
    imgs = readall(imds);
end