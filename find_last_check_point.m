% find_last_check_point 
% Find the lastest network model under the specified directory
% Input:
%   exp_dir:string
% Output:
%   epoch:int
function epoch = find_last_check_point(exp_dir)
    list = dir(fullfile(exp_dir, 'net_epoch_*.mat'));
    tokens = regexp({list.name}, 'net_epoch_([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;
end