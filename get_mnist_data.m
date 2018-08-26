function images = get_mnist_data(dataDir)
    % mnist中不仅有images还有labels，但是我们只需要用到images而已
    files = {'train-images-idx3-ubyte', ...
         't10k-images-idx3-ubyte'} ;
     
     if ~exist(dataDir, 'dir')
         mkdir(dataDir) ;
     end
     
     for i=1:numel(files)
         if ~exist(fullfile(dataDir, files{i}), 'file')
             url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
             fprintf('downloading %s\n', url) ;
             gunzip(url, dataDir) ;
         end
     end
     
     f=fopen(fullfile(dataDir, 'train-images-idx3-ubyte'),'r') ;
     images1=fread(f,inf,'uint8');
     fclose(f) ;
     images1=permute(reshape(images1(17:end),28,28,60e3),[2 1 3]) ;
     
     f=fopen(fullfile(dataDir, 't10k-images-idx3-ubyte'),'r') ;
     images2=fread(f,inf,'uint8');
     fclose(f) ;
     images2=permute(reshape(images2(17:end),28,28,10e3),[2 1 3]) ;
     
     images = reshape(cat(3, images1, images2),28,28,1,[]);
     images = single(images);
end