function [imgs] = read_images(folder)
    imds = imageDatastore(folder,"FileExtensions",".gif");
    imgs = readall(imds);
end