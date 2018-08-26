clc;
clear;

% ------- setting -----
epoch = 10;
batch_size = 64;
gpus = [1,2];
% ----------- training ---------
vl_setupnn();

generator = generator_init();
discriminator = discriminator_init();
stateD = [];
stateG = [];

real_images = get_mnist_data('./data');
real_images = real_images/255;
exp_dir = './model';
load_model_flag = true;

generator.mode = 'normal';
discriminator.mode = 'normal';

solverOpts = struct('beta1', 0.9, 'beta2', 0.999, 'eps', 1e-8);
params = struct('learningRate', 0.001, 'solver', @solver.adam, 'solverOpts', solverOpts, ...
                'gpus', gpus, 'epoch', epoch, 'batch_size', batch_size);

% if the load_model_flag is true, it will try to load trained network in
% exp_dir
if load_model_flag
    start_epoch = find_last_check_point(exp_dir);
    if start_epoch >=1
        path = model_path(exp_dir, start_epoch);
        fprintf('loading model: %s\n', path);
        [generator, discriminator, stateG, stateD] = load_model_fun(path);
    end
    start_epoch = start_epoch + 1;
else
    start_epoch = 1;
end

% if the stateD or stateG is empty, init them
% stateD and stateG are used to update network in momentum solver
if isempty(stateD) || isempty(stateG)
    stateD.solverState = cell(1, numel(discriminator.params));
    stateD.solverState(:)={0};
    stateG.solverState = cell(1, numel(generator.params));
    stateG.solverState(:)={0};
end

if numel(params.gpus)>0
    prepare_gpus(params.gpus);
end

% start to train network
for i = start_epoch:params.epoch
    if numel(params.gpus) <= 1
        [generator, discriminator, stateG, stateD] = processEpoch(generator, discriminator, stateG, stateD, i, real_images, params);
    else
        spmd
            try
                [generator, discriminator, stateG, stateD] = processEpoch(generator, discriminator, stateG, stateD, i, real_images, params);
            catch epoch_error
                epoch_error
                for k=1:numel(epoch_error.stack)
                    epoch_error.stack(k)
                end
            end
            if labindex == 1
                % save model
                path = model_path(exp_dir, i);
                save_model_fun(path, generator, discriminator, stateG, stateD);
                fprintf('save modle: %s\n', path);
            end
        end
    end
end
% With multiple GPUs, return one copy
if isa(generator, 'Composite')
    generator = generator{1} ; 
end
if isa(discriminator, 'Composite')
    discriminator = discriminator{1} ; 
end
%% processEpoch
function [generator, discriminator, stateG, stateD] = processEpoch(generator, discriminator, stateG, stateD, epoch_index, real_images, params)
    % ---------------------------
    numGpus = numel(params.gpus) ;
    if numGpus >= 1
        generator.move('gpu') ;
        discriminator.move('gpu');
        
        for k = 1:numel(stateG.solverState)
            s = stateG.solverState{k} ;
            if isnumeric(s)
                stateG.solverState{k} = gpuArray(s) ;
            elseif isstruct(s)
                stateG.solverState{k} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
        
        for k = 1:numel(stateD.solverState)
            s = stateD.solverState{k} ;
            if isnumeric(s)
                stateD.solverState{k} = gpuArray(s) ;
            elseif isstruct(s)
                stateD.solverState{k} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
    end
    
    if numGpus > 1
        parameterServer.method = 'mmap' ;
        parameterServer.prefix = 'mcn' ;
        
        parservG = ParameterServer(parameterServer) ;
        generator.setParameterServer(parservG);
        
        parservD = ParameterServer(parameterServer) ;
        discriminator.setParameterServer(parservD);
    else
        parservG = [] ;
        parservD = [];
    end
    
    % ------------------------
    % the data are divided according to batch size
    batch_count = ceil(size(real_images,4)/params.batch_size);
    for j=1:batch_count
        if j < batch_count
            batch_index_start = (j-1)* params.batch_size + 1 + (labindex-1);
            batch_index_end = j* params.batch_size;
        else
            batch_index_start = (j-1)* params.batch_size + 1 + (labindex-1);
            batch_index_end = size(real_images,4);
        end
        % gather real images, noises and lables
        batch_real_images = real_images(:,:,:,batch_index_start : numlabs : batch_index_end);
        batch_noise = single(normrnd(0, 0.1, 1, 1, 100, size(batch_real_images,4)));
        batch_real_labels = ones(1,1,1,size(batch_real_images,4), 'single');
        batch_fake_labels = -ones(1,1,1,size(batch_real_images,4), 'single');
        
        if numGpus >= 1
            batch_real_images = gpuArray(batch_real_images);
            batch_noise = gpuArray(batch_noise);
            batch_real_labels = gpuArray(batch_real_labels);
            batch_fake_labels = gpuArray(batch_fake_labels);
        end
        % ----------------------------------------
        % train discriminator firstly
        generator.eval({'noises', batch_noise});
        batch_fake_images = generator.getVar('generator_output');
        batch_fake_images = batch_fake_images.value;

        % set the accumulateParamDers to 0, the derivative of the
        % params with repects to loss will be overwrite
        discriminator.accumulateParamDers = 0;
        discriminator.eval({'images', batch_fake_images, 'labels', batch_fake_labels}, {'loss', 1}, 'holdOn', 1);
        d_loss = discriminator.getVar('loss').value;
        % set the accumlateParamsDers to 1, the derivative is equal to
        % the old one plus the new one
        discriminator.accumulateParamDers = 1;
        discriminator.eval({'images', batch_real_images, 'labels', batch_real_labels}, {'loss', 1}, 'holdOn', 0);
        d_loss = d_loss + discriminator.getVar('loss').value;
        
        % updete discriminator
        if ~isempty(parservD)
            parservD.sync();
        end
        stateD = update_network_with_gpus(discriminator, stateD, params, parservD);

        % ----------------------------------
        % train generator
        generator.eval({'noises', batch_noise});
        batch_fake_images = generator.getVar('generator_output');
        batch_fake_images = batch_fake_images.value;

        discriminator.accumulateParamDers = 0;
        discriminator.eval({'images', batch_fake_images, 'labels', batch_real_labels}, {'loss', 1}, 'holdOn', 0);
        g_loss = discriminator.getVar('loss').value;
        % the output derivative of generator is the input derivative of
        % discriminator
        der_from_discriminator = discriminator.getVar('images');
        der_from_discriminator = der_from_discriminator.der;
        
        generator.accumulateParamDers = 0;
        generator.eval({'noises', batch_noise}, {'generator_output', der_from_discriminator}, 'holdOn', 0);
        % update generator
        if ~isempty(parservG)
            parservG.sync();
        end
        stateG = update_network_with_gpus(generator, stateG, params, parservG);
        % -----------------------------------------------------
        % save sample for observe the effect of generator easily
        if mod(j,100)==0 && labindex==1
            if numGpus >= 1
                g_loss = gather(g_loss);
                d_loss = gather(d_loss);
                batch_fake_images = gather(batch_fake_images);
            end
            fprintf('g_loss:%f, d_loss:%f\n', g_loss, d_loss);
            save_path = sprintf('./pics/epoch_%d_%d.png', epoch_index, j);
            save_sample(batch_fake_images, [4,4], save_path);
            fprintf('save %s\n', save_path);
        end
    end
    
    for k = 1:numel(stateG.solverState)
        s = stateG.solverState{k} ;
        if isnumeric(s)
            stateG.solverState{k} = gather(s) ;
        elseif isstruct(s)
            stateG.solverState{k} = structfun(@gather, s, 'UniformOutput', false) ;
        end
    end
    
    for k = 1:numel(stateD.solverState)
        s = stateD.solverState{k} ;
        if isnumeric(s)
            stateD.solverState{k} = gather(s) ;
        elseif isstruct(s)
            stateD.solverState{k} = structfun(@gather, s, 'UniformOutput', false) ;
        end
    end
    
    generator.move('cpu');
    discriminator.move('cpu');
end