function net = generator_init()
    net = dagnn.DagNN();

    last_added.var = 'noises';
    last_added.channels = 100;
    [net, last_added] = add_generator_bolck(net, 'block_1', last_added, 3, 1, 0, 64);
    [net, last_added] = add_generator_bolck(net, 'block_2', last_added, 3, 2, 0, 32);
    [net, last_added] = add_generator_bolck(net, 'block_3', last_added, 5, 1, 0, 32);
    [net, last_added] = add_generator_bolck(net, 'block_4', last_added, 3, 2, 0, 32);
    [net, last_added] = add_generator_bolck(net, 'block_5', last_added, 3, 1, 0, 16);
    
    net.addLayer('convt_6',... % layer name
        dagnn.ConvTranspose('size', [4,4,1,16], 'upsample', 1, 'crop', 0, 'hasBias', true),... % layer
        last_added.var,... % input var name
        'convt_6_output',...% output var name
        {'convt_6_filters', 'convt_6_biases'}); % params name

    net.addLayer('sigmoid_layer', ...
        dagnn.Sigmoid(), ...
        'convt_6_output',...
        'generator_output');

    net.initParams();
    % set precious = 1, let the generator_output var in network will not be clean after
    % forward propagation
    net.vars(net.getVarIndex('generator_output')).precious = 1;
end

function [net, last_added] = add_generator_bolck(net, block_name, last_added, ksize, upsample, crop, out_channels)
    net.addLayer([block_name, '_convt'],...
        dagnn.ConvTranspose('size', [ksize, ksize, out_channels, last_added.channels], 'upsample', upsample, 'crop', crop, 'hasBias', true),...
        last_added.var,...
        [block_name, '_convt_output'],...
        {[block_name, '_convt_f'], [block_name, '_convt_b']});
    
    net.addLayer([block_name, '_batch_norm'], ...
        dagnn.BatchNorm('numChannels', out_channels, 'epsilon', 1e-5), ...
        [block_name, '_convt_output'], ...
        [block_name, '_batch_norm_output'], ...
        {[block_name, '_bn_w'], [block_name, '_bn_b'], [block_name, '_bn_m']}) ;

    net.addLayer([block_name, '_relu'], ...
        dagnn.ReLU('leak', 0.2), ...
        [block_name, '_batch_norm_output'],...
        [block_name, '_relu_output']);
    
    last_added.var = [block_name, '_relu_output'];
    last_added.channels = out_channels;
end