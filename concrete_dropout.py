
import numpy
np = numpy
import  scipy.special 

import theano
import theano.tensor as T
floatX = theano.config.floatX
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#rng = RandomStreams(seed=1)
# TODO: seed properly...
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=427)

import lasagne

from lasagne.objectives import categorical_crossentropy as cc
from helpers import flatten_list

from BHNs import Base_BHN
from modules import LinearFlowLayer, IndexLayer, CoupledDenseLayer, PermuteLayer
from utils import log_normal

from capy_utils import randn, tsrf


# TODO: get rid of this function and below
def concrete_dropout(p, x, eps=1e-7, rng=srng):
    eps = np.float32(eps)
    temp = 0.1
    unif_noise = rng.uniform(size=x.shape)

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
    # TODO
    #return ConcreteDropoutLayer (layer, p)

# TODO: shapes
pp = T.nnet.sigmoid(theano.shared(randn(1,3).astype('float32')*.4, broadcastable=(1,0)))
xx = tsrf(2,3)
print pp.eval()
print xx.eval()
print concrete_dropout(pp, xx).eval()










# TODO: for the original concrete dropout, we need to have a prior on the WEIGHTS not drop_probs!







class ConcreteDropoutLayer(lasagne.layers.base.Layer):
    def __init__(self, incoming, 
                    drop_probs=None, temp=0.1, eps=1e-7, srng=srng,
                    **kwargs):
        super(ConcreteDropoutLayer, self).__init__(incoming, **kwargs)
        self.__dict__.update(locals())
        self.eps = np.float32(self.eps)
        self.temp = np.float32(self.temp)

        if drop_probs is None:
            assert False # TODO

    def get_output_shape_for(self,input_shape):
        return input_shape

    # adapted from: https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout.ipynb
    def get_output_for(self, inputs):
        x = inputs
        eps = self.eps
        temp = self.temp
        unif_noise = self.srng.uniform(size=x.shape)

        # TODO: case where the drop_prob is different for different units (I think I did it now? test it!)
        smooth_dropout_mask = (
            T.log(self.drop_probs + eps)
            - T.log(1. - self.drop_probs + eps)
            + T.log(unif_noise + eps)
            - T.log(1. - unif_noise + eps)
        )
        smooth_dropout_mask = T.nnet.sigmoid(smooth_dropout_mask / temp)

        retain_prob = 1. - self.drop_probs
        x *= (1. - smooth_dropout_mask)
        x /= retain_prob
        return x




# N.B.: ws[1] --> ws[0] (i.e. we drop out INPUTS to units (starting in input space))
# TODO: we should maybe still have a prior on the magnitudes of the weights?
class MLPConcreteDropout_BHN(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of dropout params 
    parameters of MLP
    """

    
    def __init__(self,
            lbda=1, # TODO: add a prior for the weights
                 alpha=2, # alpha > beta ==> we prefer units to have high dropout probability (simplicity prior)
                 beta=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling=True,
                 n_hiddens=1,
                 n_units=200,
                 n_classes=10,
                 noise_distribution='spherical_gaussian',
                 **kargs):
        
        self.__dict__.update(locals())
        self.n_hiddens = n_hiddens
        self.n_units = n_units
        self.n_classes = n_classes
        self.weight_shapes = list()        
        self.weight_shapes.append((784,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,n_classes))
        self.num_params = sum(ws[0] for ws in self.weight_shapes)
        
        self.coupling = coupling
        #
        self.alpha = alpha
        self.beta = beta
        self.denom = scipy.special.beta(alpha,beta)
        #
        super(MLPConcreteDropout_BHN, self).__init__(lbda=-1,# TODO: shouldn't be used!
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior,
                                                **kargs)
    
    
    def _get_hyper_net(self):
        # inition random noise
        if self.noise_distribution == 'spherical_gaussian':
            self.ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        elif self.noise_distribution == 'exponential_MoG':
            self.ep = self.srng.normal(size=(self.wd1, self.num_params), dtype=floatX)
            self.ep += 2 * self.srng.binomial(size=(self.wd1, self.num_params), dtype=floatX) - 1
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
        self.logits = lasagne.layers.get_output(h_net,self.ep)
        self.drop_probs = T.nnet.sigmoid(self.logits)
        self.logdets = sum([lasagne.layers.get_output(ld,self.ep) for ld in logdets_layers])
        # TODO: test this!
        self.logdets += T.log(T.grad(T.sum(self.drop_probs), self.logits)).sum()
        self.logqw = - self.logdets
        # TODO: we should multiply this by #units if we don't output them independently...
        self.logpw = (self.alpha-1)*T.log(self.drop_probs).sum() + (self.beta-1)*T.log(1 - self.drop_probs).sum() # - np.log(self.denom) #<--- this term is constant
        # we'll compute the whole KL term right here...
        self.kl = (self.logqw - self.logpw).mean()
    
    # TODO: below
    def _get_primary_net(self):
        t = 0 #np.cast['int32'](0) # TODO: what's wrong with np.cast
        p_net = lasagne.layers.InputLayer([None,784])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            # only need ws[0] parameters 
            # just like we would rescale the incoming weights for every activation, we now drop the activations for that activation
            num_param = ws[0]
            drop_prob = self.drop_probs[:,t:t+num_param].reshape((self.wd1,num_param))
            p_net = ConcreteDropoutLayer(p_net, drop_prob, srng=self.srng)
            #
            p_net = lasagne.layers.DenseLayer(p_net,ws[1])
            #print p_net.output_shape
            t += num_param
            
        p_net.nonlinearity = lasagne.nonlinearities.softmax # replace the nonlinearity
                                                    # of the last layer
                                                    # with softmax for
                                                    # classification
        
        y = T.clip(lasagne.layers.get_output(p_net,inputs), 0.001, 0.999) # stability
        
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

    def _init_pnet(self,init_batch):
        pass


if __name__ == '__main__':
    
    
    # TODO: get rid of the last layer of concrete dropout (for the output layer)
    init_batch = np.random.rand(3,784).astype('float32') 
    model = MLPConcreteDropout_BHN(
                              perdatapoint=0,
                              prior=log_normal,
                              coupling=2,
                              n_hiddens=2,
                              n_units=32,
                              init_batch=init_batch)
    
    print model.predict_proba(init_batch)
    
    
