% model_path 
% return model path according to the exp_dir and epoch
% Input:
%   exp_dir:string, model folder
%   epoch:int
% Output:
%   path:string, model path
function path = model_path(exp_dir, epoch)
    path = fullfile(exp_dir, sprintf('net_epoch_%d.mat', epoch));
end
