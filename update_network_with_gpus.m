function state = update_network_with_gpus(net, state, params, parserv)
    numGpus = numel(params.gpus);
    numWorkers = max(1, numGpus);
    
    for p=1:numel(net.params)
        if ~isempty(parserv)
            parDer = parserv.pullWithIndex(p);
        else
            parDer = net.params(p).der ;
        end
        parDer = parDer / numWorkers;
        
        switch net.params(p).trainMethod
            case 'average'
                thisLR = net.params(p).learningRate ;
                net.params(p).value = vl_taccum(...
                    1 - thisLR, net.params(p).value, ...
                    (thisLR/net.params(p).fanout),  parDer) ;
                
            case 'gradient'
                thisLR = params.learningRate * net.params(p).learningRate ;
                if isempty(params.solver)
                    % Update parameters.
                    net.params(p).value = vl_taccum(...
                        1,  net.params(p).value, -thisLR, parDer) ;
                else
                    % call solver function to update weights
                    [net.params(p).value, state.solverState{p}] = ...
                        params.solver(net.params(p).value, state.solverState{p}, ...
                        parDer, params.solverOpts, thisLR) ;
                end
            otherwise
                error('Unknown training method ''%s'' for parameter ''%s''.', ...
                    net.params(p).trainMethod, ...
                    net.params(p).name) ;
        end
    end
end