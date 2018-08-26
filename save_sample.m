function save_sample(images, arrangement, path)
    [folder, ~, ~] = fileparts(path);
    if ~exist(folder, 'dir')
         mkdir(folder) ;
    end
    if isa(images, 'gpuArray')
        images = gather(images);
    end
    % show generated images
    sz = size(images) ;
    row = arrangement(1);
    col = arrangement(2);
    im = zeros(row*sz(1), col*sz(2), sz(3), 'uint8');
    for ii=1:row
        for jj=1:col
            idx = col*(ii-1)+jj ;
            if idx<=sz(4)
                im((ii-1)*sz(1)+1:ii*sz(1),(jj-1)*sz(2)+1:jj*sz(2),:) = uint8(images(:,:,:,idx)*255);
            end
        end
    end
    imwrite(im, path);
end