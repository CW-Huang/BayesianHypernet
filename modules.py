# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:29:14 2017

@author: Chin-Wei
"""

import theano
import theano.tensor as T
import numpy as np
floatX = theano.config.floatX
import matplotlib.pyplot as plt
import lasagne
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import get_output

conv = lasagne.theano_extensions.conv


delta = 0.001

class CoupledDenseLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, num_units, W=init.Normal(0.0001),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(CoupledDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1]/2))

        self.W1 = self.add_param(W, (num_inputs, num_units), name="cpds_W1")
        self.W21 = self.add_param(W, (num_units, num_inputs), name="cpds_W21")
        self.W22 = self.add_param(W, (num_units, num_inputs), name="cdds_W22")
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="cpds_b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (num_inputs,), name="cpds_b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (num_inputs,), name="cpds_b22",
                                      regularizable=False)
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        num_inputs = input.shape[1]
        input1 = input[:,:num_inputs/2]
        input2 = input[:,num_inputs/2:]
        output1 = input1
        
        a = T.dot(input1,self.W1)
        if self.b1 is not None:
            a = a + self.b1
        h = self.nonlinearity(a)
        
        s_ = T.dot(h,self.W21)
        if self.b21 is not None:
            s_ = s_ + self.b21
        s = T.exp(s_) + 0.001
        ls = T.log(s)
        
        m = T.dot(h,self.W22)
        if self.b22 is not None:
            m = m + self.b22
            
        output2 = s * input2 + m
        output = T.concatenate([output1,output2],1)
        
        return output, ls.sum(1)

class NormalizingPlanarFlowLayer(lasagne.layers.Layer):
    """
    Adapted from 
    https://github.com/casperkaae/parmesan/blob/master/parmesan/layers/flow.py
    
    Normalizing Planar Flow Layer as described in Rezende et
    al. [REZENDE]_ (Equation numbers and appendixes refers to this paper)
    Eq. (8) is used for calculating the forward transformation f(z).
    The last term of eq. (13) is also calculated within this layer and
    returned as an output for computational reasons. Furthermore, the
    transformation is ensured to be invertible using the constrains
    described in appendix A.1
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    u,w : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.
    References
    ----------
        ..  [REZENDE] Rezende, Danilo Jimenez, and Shakir Mohamed.
            "Variational Inference with Normalizing Flows."
            arXiv preprint arXiv:1505.05770 (2015).
    """
    def __init__(self, incoming, u=lasagne.init.Normal(),
                 w=lasagne.init.Normal(),
                 b=lasagne.init.Constant(0.0), **kwargs):
        super(NormalizingPlanarFlowLayer, self).__init__(incoming, **kwargs)
        
        num_latent = int(np.prod(self.input_shape[1:]))
        
        self.u = self.add_param(u, (num_latent,), name="u")
        self.w = self.add_param(w, (num_latent,), name="w")
        self.b = self.add_param(b, tuple(), name="b") # scalar
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    
    def get_output_for(self, input, **kwargs):
        # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
        # 2) calculate the forward transformation of the input f(z) (Eq. 8)
        # 3) calculate u_hat^T psi(z) 
        # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the LL function
        
        z = input
        # z is (batch_size, num_latent_units)
        uw = T.dot(self.u,self.w)
        muw = -1 + T.nnet.softplus(uw) # = -1 + T.log(1 + T.exp(uw))
        u_hat = self.u + (muw - uw) * T.transpose(self.w) / T.sum(self.w**2)
        zwb = T.dot(z,self.w) + self.b
        f_z = z + u_hat.dimshuffle('x',0) * lasagne.nonlinearities.tanh(zwb).dimshuffle(0,'x')
        
        psi = T.dot( (1-lasagne.nonlinearities.tanh(zwb)**2).dimshuffle(0,'x'),  self.w.dimshuffle('x',0)) # tanh(x)dx = 1 - tanh(x)**2
        psi_u = T.dot(psi, u_hat)

        logdet_jacobian = T.log(T.abs_(1 + psi_u))
        
        return [f_z, logdet_jacobian]


class CoupledWNDenseLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, num_units, W=init.Normal(1),
                 r=init.Normal(0.0001),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(CoupledWNDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1]/2))

        self.W1 = self.add_param(W, (num_inputs, num_units), name="cpds_W1")
        self.W21 = self.add_param(W, (num_units, num_inputs), name="cpds_W21")
        self.W22 = self.add_param(W, (num_units, num_inputs), name="cdds_W22")
        self.r1 = self.add_param(r, (num_units,), name='cpds_r1')
        self.r21 = self.add_param(r, (num_units,), name='cpds_r21')
        self.r22 = self.add_param(r, (num_units,), name='cpds_r22')
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="cpds_b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (num_inputs,), name="cpds_b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (num_inputs,), name="cpds_b22",
                                      regularizable=False)
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        num_inputs = input.shape[1]
        input1 = input[:,:num_inputs/2]
        input2 = input[:,num_inputs/2:]
        output1 = input1
        
        norm1 = T.sqrt(T.sum(T.square(self.W1),axis=0,keepdims=True))
        W1 = self.W1 / norm1
        norm21 = T.sqrt(T.sum(T.square(self.W21),axis=0,keepdims=True))
        W21 = self.W21 / norm21
        norm22 = T.sqrt(T.sum(T.square(self.W22),axis=0,keepdims=True))
        W22 = self.W22 / norm22
        
        a = self.r1 * T.dot(input1,W1)
        if self.b1 is not None:
            a = a + self.b1
        h = self.nonlinearity(a)
        
        s_ = self.r21 * T.dot(h,W21)
        if self.b21 is not None:
            s_ = s_ + self.b21
        s = T.exp(s_) + 0.001
        ls = T.log(s)
        
        m = self.r22 * T.dot(h,W22)
        if self.b22 is not None:
            m = m + self.b22
            
        output2 = s * input2 + m
        output = T.concatenate([output1,output2],1)
        
        return output, ls.sum(1)

class CoupledConv1DLayer(lasagne.layers.base.Layer):
    """
    shape[1] should be even number
    """
    def __init__(self, incoming, num_units, filter_size,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=True, convolution=conv.conv1d_mc0, **kwargs):
        super(CoupledConv1DLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.filter_size = filter_size
        self.num_units = num_units
        self.flip_filters = flip_filters
        self.convolution = convolution

        
        W1_shape = (num_units,1,filter_size)
        W21_shape = (1,num_units,filter_size)
        W22_shape = (1,num_units,filter_size)
        
        self.W1 = self.add_param(W, W1_shape, name="W1")
        self.W21 = self.add_param(W, W21_shape, name="W21")
        self.W22 = self.add_param(W, W22_shape, name="W22")
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (1,), name="b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (1,), name="b22",
                                      regularizable=False)
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        border_mode = 'half'
        
        num_units = self.num_units
        filter_size = self.filter_size
        W1_shape = (num_units,1,filter_size)
        W21_shape = (1,num_units,filter_size)
        W22_shape = (1,num_units,filter_size)
        num_inputs = self.input_shape[1]
        input1 = input[:,:num_inputs/2]
        input2 = input[:,num_inputs/2:]
        output1 = input1
        
        input_shape = self.input_shape 
        input1_shape = (input_shape[0], 1, num_inputs/2)
        h_shape = (input_shape[0], num_units, num_inputs/2)
        conved = self.convolution(input1.reshape((-1,1,num_inputs/2)), self.W1,
                                  input1_shape, W1_shape,
                                  subsample=(1,),
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)

        if self.b1 is not None:
            a = conved + self.b1.dimshuffle('x',0,'x')
        h = self.nonlinearity(a)
        
        conved = self.convolution(h, self.W21,
                                  h_shape, W21_shape,
                                  subsample=(1,),
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        
        if self.b21 is not None:
            s_ = conved + self.b21.dimshuffle('x',0,'x')
        s = T.nnet.softplus(s_).reshape((-1,num_inputs/2)) + delta
        ls = T.log(s)
        
        conved = self.convolution(h, self.W22,
                                  h_shape, W22_shape,
                                  subsample=(1,),
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        
        if self.b22 is not None:
            m = conved + self.b22.dimshuffle('x',0,'x')
        m = m.reshape((-1,num_inputs/2))
        
        output2 = s * input2 + m
        output = T.concatenate([output1,output2],1)

        return output, ls.sum(1)


class LinearFlowLayer(lasagne.layers.base.Layer):    
    """
    Scale and shift inputs, elementwise
    """
    def __init__(self, incoming, W=init.Normal(0.01,-7),
                 b=init.Normal(0.01,0),
                 **kwargs):
        super(LinearFlowLayer, self).__init__(incoming, **kwargs)
        
        num_inputs = int(np.prod(self.input_shape[1]))

        self.W = self.add_param(W, (num_inputs,), name="lf_W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_inputs,), name="lf_b",
                                    regularizable=False)
            
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        s = T.exp(self.W) + delta
        output = input * s
        if self.b is not None:
            output = output + self.b
        
        return output, (T.ones_like(input)*T.log(s)).sum(1)

# CC
class ConvexBiasLayer(lasagne.layers.base.Layer):    
    """
    Scale and shift inputs, elementwise
    """
    def __init__(self, incoming, W=init.Normal(0.01,-7),
                 b=init.Normal(0.01,0),
                 **kwargs):
        super(ConvexBiasLayer, self).__init__(incoming, **kwargs)
        
        num_inputs = int(np.prod(self.input_shape[1]))

        self.W = self.add_param(W, (1,), name="lf_W")
        self.b = self.add_param(b, (num_inputs,), name="lf_b",
                                regularizable=False)
            
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        s = T.nnet.sigmoid(self.W) + delta
        output = input * s + (1-s) * self.b
        
        return output, (T.ones_like(input)*T.log(s)).sum(1)


class IndexLayer(lasagne.layers.Layer):
    """
    Return the given index of input tuple
    """
    def __init__(self, incoming, index, output_shape=None, **kwargs):
        super(IndexLayer, self).__init__(incoming, **kwargs)
        self.index = index
        self.output_shape_ = output_shape
    
    def get_output_shape_for(self, input_shape):
        if  self.output_shape_ is not None:
            return self.output_shape_
        else:
            return super(IndexLayer, self).get_output_shape_for(input_shape)

    def get_output_for(self, input, **kwargs):
        return input[self.index] 


class ReverseLayer(lasagne.layers.Layer):
    """
    Reverse the order of features 
    """
    def __init__(self, incoming, num_units, axis=-1, **kwargs):
        super(ReverseLayer, self).__init__(incoming, **kwargs)
        indices = (np.arange(num_units))[::-1]
        self.indices = indices
        self.axis = axis
        
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        slc = [slice(None)] * input.ndim
        slc[self.axis] = self.indices
        return input[slc]



class PermuteLayer(lasagne.layers.Layer):
    
    def __init__(self, incoming, num_units, axis=-1, **kwargs):
        super(PermuteLayer, self).__init__(incoming, **kwargs)
        indices = np.random.permutation(np.arange(num_units))
        while np.all(indices == np.arange(num_units)):
            indices = np.random.permutation(np.arange(num_units))
        self.indices = indices
        self.axis = axis
        
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        
        slc = [slice(None)] * input.ndim
        slc[self.axis] = self.indices
        return input[slc]

class SplitLayer(lasagne.layers.Layer):
    
    def __init__(self,incoming, index, axis=-1, **kwargs):
        super(SplitLayer, self).__init__(incoming, **kwargs)
        self.index = index
        self.axis = axis
    
    def get_output_shape_for(self, input_shape):
        index = self.index
        axis = self.axis
        output_shape1 = input_shape[:axis] + \
                        (index,) + input_shape[axis+1:]
        output_shape2 = input_shape[:axis] + \
                        (input_shape[axis]-index,) + input_shape[axis+1:]
        
        return output_shape1, output_shape2
        
    def get_output_for(self, input, **kwargs):
        index = self.index
        axis = self.axis
        slc1 = [slice(None)] * input.ndim
        slc1[axis] = np.arange(index)
        slc2 = [slice(None)] * input.ndim
        slc2[axis] = np.arange(index,self.input_shape[axis])
        output1 = input[slc1]
        output2 = input[slc2]
        return output1, output2

    
class stochasticDenseLayer(lasagne.layers.base.MergeLayer):
    
    def __init__(self, incomings, num_units, 
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(stochasticDenseLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="stocds_b",
                                    regularizable=False)
                                    
    def get_output_shape_for(self,input_shapes):
        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]
        try:
            return (input_shape[0], weight_shape[2])
        except:
            return (input_shape[0], weight_shape[1])
        
    def get_output_for(self, inputs, **kwargs):
        """
        inputs[0].shape = (None, num_inputs)
        inputs[1].shape = (None/1, num_inputs, num_units)
        """
        input = inputs[0]
        W = inputs[1]
        activation = T.sum(input.dimshuffle(0,1,'x') * W, axis = 1)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)

    

class stochasticDenseLayer2(lasagne.layers.base.MergeLayer):
    """
    stochastic dense layer with weightnorm reparameterization
    noise on preactivation rescaling parameters
    """
    def __init__(self, incomings, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(stochasticDenseLayer2, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        input_shape = incomings[0].output_shape
        if num_leading_axes >= len(input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "leaving no trailing axes for the dot product." %
                    (num_leading_axes, len(input_shape)))
        elif num_leading_axes < -len(input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "requesting more trailing axes than there are input "
                    "dimensions." % (num_leading_axes, len(input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in input_shape[num_leading_axes:]):
            raise ValueError(
                    "A DenseLayer requires a fixed input shape (except for "
                    "the leading axes). Got %r for num_leading_axes=%d." %
                    (input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(input_shape[num_leading_axes:]))
        
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)
                                    
    def get_output_shape_for(self,input_shapes):
        return input_shapes[0][:self.num_leading_axes] + (self.num_units,)
        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]
        return (input_shape[0], weight_shape[2])
        
    def get_output_for(self, inputs, **kwargs):
        """
        inputs[0].shape = (None, num_inputs)
        inputs[1].shape = (None/1, num_units)
        """
        input = inputs[0]
        r = inputs[1]
        norm = T.sqrt(T.sum(T.square(self.W),axis=0,keepdims=True))
        W = self.W / norm
        
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        activation = T.dot(input, W) * r
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)



def stochasticConv2DLayer(incomings, num_filters, filter_size, stride=(1, 1),
                          pad=0, untie_biases=False, W=init.GlorotUniform(), 
                          b=init.Constant(0.),
                          nonlinearity=nonlinearities.rectify, 
                          flip_filters=True,
                          convolution=T.nnet.conv2d, **kwargs):
    
    incoming = incomings[0]
    W = incomings[1]
    
    layer = lasagne.layers.Conv2DLayer(
        incoming, num_filters, filter_size, stride,
        pad, untie_biases, W, b,
        nonlinearity, flip_filters,
        convolution
    )
    
    layer.W = W

    return layer
    
        

if __name__ == '__main__':
    
    """
    an example of using invertible transformation to fit a complicated 
    density function that is hard to sample from
    """
    
    def U(Z):
        """ Toroid """
        z1 = Z[:, 0]
        z2 = Z[:, 1]
        R = 5.0
        return - 2*(R-(z1**2+.5*z2**2)**0.5)**2 
        
    
    print 'building model'
    logdets_layers = []
    layer = lasagne.layers.InputLayer([None,2])
    
    layer_temp = LinearFlowLayer(layer)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    layer_temp = CoupledDenseLayer(layer,100)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    layer = PermuteLayer(layer,2)
    
    layer_temp = CoupledDenseLayer(layer,100)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    
    ep = T.matrix('ep')
    z = lasagne.layers.get_output(layer,ep)
    logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
    
    logq = - logdets
    logp = U(z)
    losses = logq - logp
    loss = losses.mean()
    
    params = lasagne.layers.get_all_params(layer)
    updates = lasagne.updates.adam(loss,params,0.001)
    
    train = theano.function([ep],loss,updates=updates)
    
    z0 = (ep - params[1]) / T.exp(params[0])
    logq_ = sum([get_output(logdet,z0) for logdet in logdets_layers])
    samples = theano.function([ep],z)

    print 'starting training'
    for i in range(20000):
        spl = np.random.randn(128,2).astype(floatX)
        l = train(spl)
    
        if i%1000==0:
            print l
    
    print "\nvisualizing"
    prior_noise = T.matrix('prior_noise')
    density = U(prior_noise)
    f0 = theano.function([prior_noise],density)
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1,2,1)
    x = np.linspace(-10,10,1000)
    y = np.linspace(-10,10,1000)
    xx,yy = np.meshgrid(x,y)
    X = np.concatenate((xx.reshape(1000000,1),yy.reshape(1000000,1)),1)
    X = X.astype(floatX)
    Z = f0(X).reshape(1000,1000)
    ax.pcolormesh(xx,yy,np.exp(Z))
    ax.axis('off')
    
    ax = fig.add_subplot(1,2,2)
    Z0 = spl = np.random.randn(100000,2).astype(floatX)
    Zs = samples(Z0)
    XX = Zs[:,0]
    YY = Zs[:,1]
    plot = ax.hist2d(XX,YY,100)
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.axis('off')
    
    
    #plt.savefig('autoregressive_ex_toroid.jpg')







