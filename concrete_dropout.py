
import  scipy.special 
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# TODO
rng = RandomStreams(seed=1)

# adapted from: https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout.ipynb
def concrete_dropout(p, x, eps=1e-7, rng=rng):
    '''
    p - dropout probability for x 
    x - activations
    '''
    eps = np.float32(eps)
    temp = 0.1
    unif_noise = rng.uniform(shape=x.shape)

    # TODO: case where the drop_prob is different for different units
    smooth_dropout_mask = (
        T.log(p + eps)
        - T.log(1. - p + eps)
        + T.log(unif_noise + eps)
        - T.log(1. - unif_noise + eps)
    )
    smooth_dropout_mask = T.nnet.sigmoid(smooth_dropout_mask / temp)

    retain_prob = 1. - p
    x *= (1. - smooth_dropout_mask)
    x /= retain_prob
    return x


class MLPConcreteDropout_BHN(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of dropout params 
    parameters of MLP
    """

    
    def __init__(self,
                 alpha=2, # alpha > beta ==> we prefer units to have high dropout probability (simplicity prior)
                 beta=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling=True,
                 n_hiddens=1,
                 n_units=200,
                 n_classes=10,
                 **kargs):
        
        self.n_hiddens = n_hiddens
        self.n_units = n_units
        self.n_classes = n_classes
        self.weight_shapes = list()        
        self.weight_shapes.append((784,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,n_classes))
        self.num_params = sum(ws[1] for ws in self.weight_shapes)
        
        self.coupling = coupling
        super(MLPConcreteDropout_BHN, self).__init__(lbda=-1,# TODO: shouldn't be used!
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior,
                                                **kargs)
        self.alpha = alpha
        self.beta = beta
        self.denom = scipy.special.beta(alpha,beta)
    
    
    def _get_hyper_net(self):
        # inition random noise
        ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        if self.coupling:
            layer_temp = CoupledDenseLayer(h_net,200)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            for c in range(self.coupling-1):
                h_net = PermuteLayer(h_net,self.num_params)
                
                layer_temp = CoupledDenseLayer(h_net,200)
                h_net = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
        
        self.h_net = h_net
        self.logits = lasagne.layers.get_output(h_net,ep)
        self.drop_probs = T.nnet.sigmoid(self.logits)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
        # TODO: test this!
        self.logdets += T.log(T.grad(T.sum(self.drop_probs)), self.logits)).sum()
        self.logqw = - logdets
        # TODO: we should multiply this by #units if we don't output them independently...
        self.logpw = (self.alpha-1)*T.log(self.drop_probs).sum() + (self.beta-1)*T.log(1 - self.drop_probs).sum() )# - np.log(self.denom) #<--- this term is constant
        # we'll compute the whole KL term right here...
        self.kl = (self.logqw - self.logpw).mean()
    
    # TODO: below
    def _get_primary_net(self):
        t = 0 #np.cast['int32'](0) # TODO: what's wrong with np.cast
        p_net = lasagne.layers.InputLayer([None,784])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            # only need ws[1] parameters 
            # just like we would rescale the incoming weights for every activation, we now drop the activations for that activation
            num_param = ws[1]
            drop_prob = self.drop_probs[:,t:t+num_param].reshape((self.wd1,ws[1]))
            p_net = lasagne.layers.DenseLayer(p_net,ws[1])
            p_net = concrete_dropout(drop_prob, p_net)
            print p_net.output_shape
            t += num_param
            
        p_net.nonlinearity = nonlinearities.softmax # replace the nonlinearity
                                                    # of the last layer
                                                    # with softmax for
                                                    # classification
        
        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    # TODO: do these need to be modified at all??
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))

    def _get_elbo(self):
        """
        negative elbo, an upper bound on NLL
        """
        self.logpyx = - cc(self.y,self.target_var).mean()
        self.loss = - (self.logpyx - \
                       self.weight * self.kl/T.cast(self.dataset_size,floatX))

        # DK - extra monitoring
        params = self.params
        ds = self.dataset_size
        self.logpyx_grad = flatten_list(T.grad(-self.logpyx, params, disconnected_inputs='warn')).norm(2)
        self.logpw_grad = flatten_list(T.grad(-self.logpw.mean() / ds, params, disconnected_inputs='warn')).norm(2)
        self.logqw_grad = flatten_list(T.grad(self.logqw.mean() / ds, params, disconnected_inputs='warn')).norm(2)
        self.monitored = [self.logpyx, self.logpw, self.logqw,
                          self.logpyx_grad, self.logpw_grad, self.logqw_grad]
        
        self.logpyx = - cc(self.y,self.target_var).mean()
        self.loss = - (self.logpyx - \
                       self.weight * self.kl/T.cast(self.dataset_size,floatX))
