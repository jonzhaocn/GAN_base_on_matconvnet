clc;
clear;

% ------- setting -----
epoch = 10;
batch_size = 64;
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
params = struct('learningRate', 0.001, 'solver', @solver.adam, 'solverOpts', solverOpts);

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

% start to train network
for i = start_epoch:epoch
    % the data are divided according to batch size
    batch_count = ceil(size(real_images,4)/batch_size);
    for j=1:batch_count
        if j < batch_count
            batch_index_start = (j-1)*batch_size + 1;
            batch_index_end = j*batch_size;
        else
            batch_index_start = (j-1)*batch_size + 1;
            batch_index_end = size(real_images,4);
        end
        % gather real images, noises and lables
        batch_real_images = real_images(:,:,:,batch_index_start:batch_index_end);
        batch_noise = single(normrnd(0, 0.1, 1, 1, 100, size(batch_real_images,4)));
        batch_real_labels = ones(1,1,1,size(batch_real_images,4), 'single');
        batch_fake_labels = -ones(1,1,1,size(batch_real_images,4), 'single');
        
        % ----------------------------------------
        % train discriminator firstly
        generator.eval({'noises', batch_noise});
        batch_fake_images = generator.getVar('generator_output');
        batch_fake_images = batch_fake_images.value;
        
        % set the accumulateParamDers to 0, the derivative of the
        % params with repects to loss will be overwrite
        discriminator.accumulateParamDers = 0;
        discriminator.eval({'images', batch_fake_images, 'labels', batch_fake_labels}, {'loss', 1});
        d_loss = discriminator.getVar('loss').value;
        % set the accumlateParamsDers to 1, the derivative is equal to
        % the old one plus the new one
        discriminator.accumulateParamDers = 1;
        discriminator.eval({'images', batch_real_images, 'labels', batch_real_labels}, {'loss', 1});
        d_loss = d_loss + discriminator.getVar('loss').value;
        % updete discriminator
        stateD = update_network(discriminator, stateD, params);
        % ----------------------------------
        % train generator
        generator.eval({'noises', batch_noise});
        batch_fake_images = generator.getVar('generator_output');
        batch_fake_images = batch_fake_images.value;
        
        discriminator.accumulateParamDers = 0;
        discriminator.eval({'images', batch_fake_images, 'labels', batch_real_labels}, {'loss', 1});
        g_loss = discriminator.getVar('loss').value;
        % the output derivative of generator is the input derivative of
        % discriminator
        der_from_discriminator = discriminator.getVar('images');
        der_from_discriminator = der_from_discriminator.der;
        
        generator.accumulateParamDers = 0;
        generator.eval({'noises', batch_noise}, {'generator_output', der_from_discriminator});
        % update generator
        stateG = update_network(generator, stateG, params);
        % -----------------------------------------------------
        % save sample for observe the effect of generator easily
        if mod(j,100)==0
            fprintf('g_loss:%f, d_loss:%f\n', g_loss, d_loss);
            save_path = sprintf('./pics/epoch_%d_%d.png', i, j);
            save_sample(batch_fake_images, [4,4], save_path);
            fprintf('save %s\n', save_path);
        end
        % ---------------------------------------------------------
        % save model
        if j==batch_count
            path = model_path(exp_dir, i);
            save_model_fun(path, generator, discriminator, stateG, stateD);
            fprintf('save modle: %s\n', path);
        end
    end
end