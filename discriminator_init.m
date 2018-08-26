function net = discriminator_init()
    net = dagnn.DagNN();
    
    last_added.channels = 1;
    last_added.var = 'images';
    
    [net, last_added] = add_discriminator_block(net, 'block_1', last_added, 3, 1, 0, 1, 16);
    [net, last_added] = add_discriminator_block(net, 'block_2', last_added, 3, 2, 0, 1, 32);
    [net, last_added] = add_discriminator_block(net, 'block_3', last_added, 3, 1, 0, 1, 32);
    [net, last_added] = add_discriminator_block(net, 'block_4', last_added, 3, 2, 0, 1, 16);
    
    net.addLayer('conv_5_layer',... % layer name
        dagnn.Conv('size', [4,4,last_added.channels,1], 'stride', 1, 'pad', 0, 'dilate', 1, 'hasBias', true),... % layer
        last_added.var,... % input var name
        'conv_5_output',... % output var name
        {'conv_5_filters', 'conv_5_biases'}); % params name
    
    net.addLayer('loss_layer',...
        dagnn.Loss('loss', 'logistic'), ...
        {'conv_5_output', 'labels'},...
        'loss');
    
    net.initParams();
    % set precious = 1, let the loss var in network will not be clean after
    % forward propagation
    net.vars(net.getVarIndex('conv_5_output')).precious = 1;
    net.vars(net.getVarIndex('loss')).precious = 1;
end

function [net, last_added] = add_discriminator_block(net, block_name, last_added, ksize, stride, pad, dilate, out_channels)
    net.addLayer([block_name, '_conv'],...
        dagnn.Conv('size', [ksize, ksize, last_added.channels, out_channels], 'stride', stride, 'pad', pad, 'dilate', dilate, 'hasBias', true),...
        last_added.var,...
        [block_name, '_conv_output'],...
        {[block_name, '_conv_f'], [block_name, '_conv_b']});
    
    net.addLayer([block_name, '_batch_norm'], ...
        dagnn.BatchNorm('numChannels', out_channels, 'epsilon', 1e-5), ...
        [block_name, '_conv_output'], ...
        [block_name, '_batch_norm_output'], ...
        {[block_name, '_bn_w'], [block_name, '_bn_b'], [block_name, '_bn_m']}) ;
    
    net.addLayer([block_name, '_relu'], ...
        dagnn.ReLU('leak', 0.2), ...
        [block_name, '_batch_norm_output'],...
        [block_name, '_relu_output']);
    
   last_added.var = [block_name, '_relu_output'];
   last_added.channels = out_channels;
end