% load_model_fun 
% load trained network model
% Input:
%   path: 
% Output:
%   generator:
%   discriminator:
%   stateG:
%   stateD:
function [generator, discriminator, stateG, stateD] = load_model_fun(path)
    load(path, 'generator', 'discriminator', 'stateG', 'stateD');
    generator = dagnn.DagNN.loadobj(generator);
    discriminator = dagnn.DagNN.loadobj(discriminator);
end