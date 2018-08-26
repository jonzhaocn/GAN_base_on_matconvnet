% save_model_fun 
% save trained network model
% Input: 
%   path:string, folder
%   generator_: dagnn.DagNN
%   discriminator_:dagnn.DagNN
%   stateG: cell array, save auxiliary variables of generator's network parameter for network update calculation
%   stateD: cell array, save auxiliary variables of discriminator's network parameter for network update calculation
% Output:
%   None
function save_model_fun(path, generator_, discriminator_, stateG, stateD)
    [folder, ~, ~] = fileparts(path);
    if ~exist(folder, 'dir')
        mkdir(folder) ;
    end
    generator = generator_.saveobj();
    discriminator = discriminator_.saveobj();
    save(path, 'generator', 'discriminator', 'stateG', 'stateD');
end