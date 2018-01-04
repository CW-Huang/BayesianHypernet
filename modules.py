# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:29:14 2017

@author: Chin-Wei
"""

import theano
import theano.tensor as T
import numpy as np
floatX = theano.config.floatX
#import matplotlib.pyplot as plt
import lasagne
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import get_output
from lasagne.layers import Conv2DLayer, DenseLayer
from lasagne.nonlinearities import rectify
from lasagne.objectives import categorical_crossentropy as cc
from lasagne.objectives import squared_error as se

from theano.tensor.var import TensorVariable as tv

from utils import log_normal
from externals_modules.made_modules import MADE

conv = lasagne.theano_extensions.conv
softplus = lambda x: T.nnet.softplus(x) + delta
exp = lambda x: T.exp(x) + delta

delta = 0.001

# DK
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=427)

# TODO: separate primary/hnet layers!
    


class CoupledDenseLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, num_units, W=init.Normal(0.0001),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(CoupledDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs1 = int(self.input_shape[1]/2)
        num_inputs2 = self.input_shape[1] - num_inputs1

        self.W1 = self.add_param(W, (num_inputs1, num_units), name="cpds_W1")
        self.W21 = self.add_param(W, (num_units, num_inputs2), name="cpds_W21")
        self.W22 = self.add_param(W, (num_units, num_inputs2), name="cdds_W22")
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="cpds_b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (num_inputs2,), name="cpds_b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (num_inputs2,), name="cpds_b22",
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


class CoupledWNDenseLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, num_units,
                 W=init.Normal(0.0001),
                 r=init.Normal(0.0001),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(CoupledWNDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        
        self.num_units = num_units
        
        num_inputs1 = int(self.input_shape[1]/2)
        num_inputs2 = self.input_shape[1] - num_inputs1


        self.W1 = self.add_param(W, (num_inputs1, num_units), name="cpds_W1")
        self.W21 = self.add_param(W, (num_units, num_inputs2), name="cpds_W21")
        self.W22 = self.add_param(W, (num_units, num_inputs2), name="cdds_W22")
        self.r1 = self.add_param(r, (num_units,), name='cpds_r1')
        self.r21 = self.add_param(r, (num_inputs2,), name='cpds_r21')
        self.r22 = self.add_param(r, (num_inputs2,), name='cpds_r22')
        if b is None:
            self.b1 = None
            self.b21 = None
            self.b22 = None
        else:
            self.b1 = self.add_param(b, (num_units,), name="cpds_b1",
                                     regularizable=False)
            self.b21 = self.add_param(b, (num_inputs2,), name="cpds_b21",
                                      regularizable=False)
            self.b22 = self.add_param(b, (num_inputs2,), name="cpds_b22",
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





def get_wn_params(P,add_param,specs,name,d1,d2):
    u,g,b = specs
    P['u_{}'.format(name)] = add_param(u,(d1,d2))   
    P['g_{}'.format(name)] = add_param(g,(d2,))     
    P['b_{}'.format(name)] = add_param(b,(d2,),
                                       regularizable=False)        

def weightnorm(W_,g):
    if W_.ndim==4:
        W_axes_to_sum = (1,2,3)
        W_dimshuffle_args = [0,'x','x','x']
    else:
        W_axes_to_sum = 0
        W_dimshuffle_args = ['x',0]
        
    norm = T.sqrt(T.sum(T.square(W_),axis=W_axes_to_sum,keepdims=True))
    W_normed = W_ / norm
    return W_normed * g.dimshuffle(*W_dimshuffle_args)

    
def dot_():
    dot = lambda X,W,b: T.dot(X,W) + b.dimshuffle('x',0)
    return dot

    
def weightnormdot(X,W,g,b,operator=dot_(),nonl=None):
    h = operator(X,weightnorm(W,g),b)
    if nonl is not None:
        return nonl(h)
    else:
        return h
        
        
class IAFDenseLayer(lasagne.layers.base.Layer):    
    def __init__(self, incoming, num_units, 
                 num_hids=1, L=1, cond_bias=False,
                 W=init.Normal(0.0001),
                 r=init.Normal(0.0001),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(IAFDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        
        
        aux_input = T.matrix()
        masks = list()      
        P = dict()
        name = 'iaf'
        for l in range(L):
            made = MADE(aux_input,incoming.input_shape[1],
                        hidden_sizes=[num_units,]*num_hids,
                        hidden_activation=None,
                        random_seed=1234+l)
            
            
            made.shuffle('Once')
            made.shuffle('Full')
            
            # saving the masks generated by `MADE`
            masks.append([m.get_value() for m in made.masks])
            # initializing parameters, 
            # # last one is mean
            for h,m in enumerate(masks[l]):
                d1,d2 = m.shape
                get_wn_params(P,self.add_param,[W,r,b],
                              '{}_l{}h{}'.format(name,l,h),d1,d2)
            # # std
            get_wn_params(P,self.add_param,[W,r,b],
                          '{}_l{}h{}s'.format(name,l,h),d1,d2)
            # # additional conditional bias
            if cond_bias:
                d1,d2 = masks[l][0]
                get_wn_params(P,self.add_param,[W,r,b],
                              '{}_l{}cb1'.format(name,l),d1,d2)
                if l != L-1:
                    get_wn_params(P,self.add_param,[W,r,b],
                                  '{}_l{}cb2'.format(name,l),d1,d2)
        
        self.P = P
        self.cond_bias = cond_bias
        self.num_hids = num_hids
        self.L = L
        self.masks = masks
        
            
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, cond_bias=None, **kwargs):
        z = input
        zs = list()
        zs.append(input)
        ss = T.zeros((input.shape[0],)) # logdet jacobian
    

        masks = self.masks        
        P = self.P
        L = self.L
        num_hids = self.num_hids
        #cond_bias = self.cond_bias
        nonl = self.nonlinearity
        name = 'iaf'
        
        for l in range(L):
            hidden = zs[l]
            for h in range(num_hids+1):
                mask = masks[l][h]
                u = P['u_{}_l{}h{}'.format(name,l,h)]
                g = P['g_{}_l{}h{}'.format(name,l,h)]
                b = P['b_{}_l{}h{}'.format(name,l,h)]
                u_ = T.switch(mask,u,0)
                if h != num_hids:
                    hidden = weightnormdot(hidden,u_,g,b,nonl=nonl)
                else:
                    mean = weightnormdot(hidden,u_,g,b,nonl=None)
            
                if h == 0 and cond_bias is not None:
                    u = P['u_{}_l{}cb1'.format(name,l,h)]
                    g = P['g_{}_l{}cb1'.format(name,l,h)]
                    b = P['b_{}_l{}cb1'.format(name,l,h)]
                    hidden_cb = weightnormdot(cond_bias,u,g,b,nonl=nonl)
                    hidden += hidden_cb
                    if l != L-1:
                        u = P['u_{}_l{}cb2'.format(name,l,h)]
                        g = P['g_{}_l{}cb2'.format(name,l,h)]
                        b = P['b_{}_l{}cb2'.format(name,l,h)]
                        cond_bias = weightnormdot(hidden_cb,u,g,b,nonl=nonl)
                        
            
            u = P['u_{}_l{}h{}s'.format(name,l,h)]
            g = P['g_{}_l{}h{}s'.format(name,l,h)]
            b = P['b_{}_l{}h{}s'.format(name,l,h)]
            u_ = T.switch(mask,u,0)
            std = weightnormdot(hidden,u_,g,b,nonl=exp)
            
            z = mean + std * zs[l]
            zs.append(z)
            ss += T.sum(T.log(std),1)
        
        return z, ss
        
        
        
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
    
        
# TODO: implement MNF (done?)
class MNFLayer(lasagne.layers.base.MergeLayer):
    
    def __init__(self, incomings, num_units, 
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 srgn=srng,
                 W_mu=lasagne.init.Normal(0.05), 
                 W_sigma=lasagne.init.Normal(0.05), 
                 num_leading_axes=1, **kwargs):
        super(MNFLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        self.srng = srng
        self.W_mu = W_mu
        self.W_sigma = W_sigma
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="stocds_b",
                                    regularizable=False)
                                    
    # TODO: rm?
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
        inputs[1].shape = (None, num_inputs)
        """
        input = inputs[0]
        Z = inputs[1] # Z_{T_f}
        activation = input * Z
        mu = T.sum(activation.dimshuffle(0,1,'x') * self.W_mu, axis = 1)
        sig = T.sum((input**2).dimshuffle(0,1,'x') * self.W_sigma, axis = 1)**.5
        ep = self.srng.normal(size=var.shape,dtype=floatX)
        activation = mu + ep * std
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation), mu, sig










class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), 
                 g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), 
                 nonlinearity=nonlinearities.rectify, **kwargs):
                     
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if type(g) == tv:
            self.g = g
        elif g is not None:
            self.g = self.add_param(g, (k,), name="g")
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            W_axes_to_sum = (1,2,3)
            W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = self.g * incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum)).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))        

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            stdv = T.sqrt(T.mean(T.square(input),axis=self.axes_to_sum))
            input /= stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m/stdv), (self.g, self.g/stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)
        
def stochastic_weight_norm(layer, weight, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
        
    layer_out = WeightNormLayer(layer, g = weight,
                                nonlinearity=nonlinearity, **kwargs)     
    return layer_out


        
        
# new BHN with WN/BN
        
from lasagne.layers import InputLayer, ElemwiseSumLayer, BatchNormLayer, \
                           MergeLayer, get_all_layers, Layer
from lasagne import utils
from difflib import get_close_matches
from warnings import warn

def NVP_dense_layer(incoming, 
                    num_units=200,
                    L=2,
                    W=init.Normal(0.0001),
                    r=init.Normal(0.0001),
                    b=init.Constant(0.), 
                    nonlinearity=nonlinearities.rectify,):
    
    layer = incoming
    shape = layer.output_shape[1]
    logdets_layers = list()
    
    for c in range(L):
       layer = PermuteLayer(layer,shape)
       layer_temp = CoupledWNDenseLayer(layer,num_units,
                                        W=W,r=r,b=b,nonlinearity=nonlinearity)
       layer = IndexLayer(layer_temp,0)
       logdets_layers.append(IndexLayer(layer_temp,1))
                    
    return layer, logdets_layers

def IAF_dense_layer(incoming, 
                    num_units=200,
                    L=2,
                    num_hids=1,
                    W=init.Normal(0.0001),
                    r=init.Normal(0.0001),
                    b=init.Constant(0.), 
                    nonlinearity=nonlinearities.rectify):

    layer_temp = IAFDenseLayer(incoming,num_units,num_hids,L=L,cond_bias=False)
    layer = IndexLayer(layer_temp,0)
    logdets_layers = IndexLayer(layer_temp,1)
    
    return layer, [logdets_layers,]


normalizable_layers = [DenseLayer,Conv2DLayer]
nlb = lambda layer: any([isinstance(layer,nl) for nl in normalizable_layers])

def hypernet(net, 
             hidden_size=512, 
             layers=0, 
             flow='IAF', 
             output_size=None,
             nlb = nlb,
             copies = 1, # 2 for bias + scale; 1 for scale only
             **kargs):

    if output_size is None:
        all_layers = lasagne.layers.get_all_layers(net)
        output_size = sum([layer.output_shape[1] for layer in all_layers
                           if not isinstance(layer,InputLayer) and 
                              not isinstance(layer,ElemwiseSumLayer) and
                              nlb(layer)]) * copies
    
    logdets_layers = []
    layer = InputLayer(shape=(None,output_size))
    layer_temp = LinearFlowLayer(layer)
    layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    if layers > 0:
        if flow == 'RealNVP':
            layer, ld_layers = NVP_dense_layer(layer, hidden_size,
                                               layers, **kargs)                        
        elif flow == 'IAF':
            layer, ld_layers = IAF_dense_layer(layer, hidden_size,
                                               layers, **kargs)        
        logdets_layers = logdets_layers + ld_layers
        
    return layer, ElemwiseSumLayer(logdets_layers), output_size
    

def slicing(params, start_index, size):
    end_index = start_index + size
    return params[:,start_index:end_index], end_index
    
def N_get_output(layer_or_layers, inputs, hnet, input_h, 
                 deterministic=False, norm_type='BN', 
                 static_bias=None, nlb=nlb, 
                 test_time=False,
                 **kwargs):


    # check if the keys of the dictionary are valid
    if isinstance(inputs, dict):
        for input_key in inputs.keys():
            if (input_key is not None) and (not isinstance(input_key, Layer)):
                raise TypeError("The inputs dictionary keys must be"
                                " lasagne layers not %s." %
                                type(input_key))
    # track accepted kwargs used by get_output_for
    accepted_kwargs = {'deterministic'}
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, utils.as_theano_expression(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = utils.as_theano_expression(inputs)
            
            
    N_params = lasagne.layers.get_output(hnet,input_h)
    index = 0
    if static_bias is not None:
        index_b = 0
        if static_bias.ndim == 1:
            static_bias = static_bias.dimshuffle('x',0)
            
    # update layer-to-expression mapping by propagating the inputs
    #last_layer = all_layers[-1]
    for layer in all_layers:
        
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)

            #if layer is not last_layer and not isinstance(layer,
            #                                              ElemwiseSumLayer):
            if not isinstance(layer, ElemwiseSumLayer) and \
               nlb(layer):
                
                nonlinearity = getattr(layer, 'nonlinearity', None)
                if nonlinearity is not None:
                    layer.nonlinearity = lambda x: x
                else:
                    nonlinearity = lambda x: x
                    
                if hasattr(layer, 'b') and layer.b is not None:
                    del layer.params[layer.b]
                    layer.b = None

                size = layer.output_shape[1]
                print size
                if norm_type == 'BN':
                    N_layer = BatchNormLayer(layer,beta=None,gamma=None)
                    layer_output = layer.get_output_for(layer_inputs, **kwargs)
                    if test_time:
                        N_output = N_layer.get_output_for(layer_output,
                                                          True)
                    else:
                        N_output = N_layer.get_output_for(layer_output,
                                                          deterministic)
                        
                elif norm_type == 'WN':
                    N_layer = WeightNormLayer(layer,b=None,g=None)
                    layer_output = layer.get_output_for(layer_inputs, **kwargs)
                    N_output = N_layer.get_output_for(layer_output,
                                                      deterministic)
                else:
                    raise Exception('normalization method {} not ' \
                                    'supported.'.format(norm_type))
                                    
                
                gamma, index = slicing(N_params,index,size)
                if static_bias is None:
                    beta, index = slicing(N_params,index,size)
                else:
                    beta, index_b = slicing(static_bias,index_b,size)
                if len(layer.output_shape) == 4:
                    gamma = gamma.dimshuffle(0,1,'x','x')
                    beta = beta.dimshuffle(0,1,'x','x')
    
                CN_output = gamma * N_output + beta
                    
                all_outputs[layer] = nonlinearity(CN_output)
                layer.nonlinearity = nonlinearity
            else:
                all_outputs[layer] = layer.get_output_for(layer_inputs, 
                                                          **kwargs)
            try:
                accepted_kwargs |= set(utils.inspect_kwargs(
                        layer.get_output_for))
            except: # TypeError:
                # If introspection is not possible, skip it
                pass
            accepted_kwargs |= set(layer.get_output_kwargs)
    
    hs = hnet.output_shape[1]
    errmsg = 'mismatch: hnet output ({}) cbn params ({})'.format(hs,index)
    assert hs == index, errmsg
    unused_kwargs = set(kwargs.keys()) - accepted_kwargs
    if unused_kwargs:
        suggestions = []
        for kwarg in unused_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions))
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def get_elbo(pred,
             targ,
             weights,
             logdets,
             weight,
             dataset_size,
             prior=log_normal,
             lbda=0,
             output_type = 'categorical'):
    """
    negative elbo, an upper bound on NLL
    """

    logqw = - logdets
    """
    originally...
    logqw = - (0.5*(ep**2).sum(1)+0.5*T.log(2*np.pi)*num_params+logdets)
        --> constants are neglected in this wrapperfrom utils import log_laplace
    """
    logpw = prior(weights,0.,-T.log(lbda)).sum(1)
    """
    using normal prior centered at zero, with lbda being the inverse 
    of the variance
    """
    kl = (logqw - logpw).mean()
    if output_type == 'categorical':
        logpyx = - cc(pred,targ).mean()
    elif output_type == 'real':
        logpyx = - se(pred,targ).mean() # assume output is a vector !
    else:
        assert False
    loss = - (logpyx - weight * kl/T.cast(dataset_size,floatX))

    return loss, [logpyx, logpw, logqw]
    
        

if __name__ == '__main__':

    if 0:
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
        
        
        if 0:    
            layer_temp = CoupledDenseLayer(layer,100)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            layer = PermuteLayer(layer,2)
            
            layer_temp = CoupledDenseLayer(layer,100)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            layer_temp = IAFDenseLayer(layer,100,1,
                                       L=2,cond_bias=False)
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
        
        """
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
        """
        
        
        #plt.savefig('autoregressive_ex_toroid.jpg')
    
    
    if 1:
        DIM_C=3
        DIM_X=16
        DIM_Y=16
        
        
        X_in = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y))
        layer = Conv2DLayer(incoming=X_in, num_filters=128, filter_size=4, 
                            stride=2, pad=1, nonlinearity=rectify)
        print layer.output_shape
        layer = Conv2DLayer(incoming=layer, num_filters=64, filter_size=4, 
                            stride=2, pad=1, nonlinearity=rectify)
        print layer.output_shape
        layer = Conv2DLayer(incoming=layer, num_filters=32, filter_size=4, 
                            stride=2, pad=1, nonlinearity=rectify)

        print layer.output_shape
        layer = DenseLayer(layer, 32, 
                           nonlinearity=rectify)
        print layer.output_shape
        layer = DenseLayer(layer, 2, 
                           nonlinearity=lasagne.nonlinearities.softmax)
        print layer.output_shape
        
        
        
    
        #input_var = T.tensor4('input_var')
        input_var = T.as_tensor_variable(
            np.random.rand(5,3,16,16).astype(np.float32)    
        )

        
        from theano.tensor.shared_randomstreams import RandomStreams
        srng = RandomStreams(seed=427)


        if 0:
            print 'example: conditioning bias'        
            hnet, ld, num_params = hypernet(layer, 100, 2, copies = 2)
        
            ep = srng.normal(size=(1,num_params),dtype=floatX)
            output_var = N_get_output(layer,input_var,hnet,ep)
            print output_var.eval().shape

        else:    
            print 'example: not conditioning bias'  
            hnet, ld, num_params = hypernet(layer, 100, 2, copies = 1, 
                                            flow='RealNVP')
    
            ep = srng.normal(size=(1,num_params),dtype=floatX)        
            static_bias = theano.shared(np.zeros((num_params)).astype('float32'))
            ### remember to concatenate this with params ### 
            
            output_var = N_get_output(layer,input_var,hnet,ep,
                                      static_bias=static_bias,
                                      norm_type='BN')
            print output_var.eval().shape



        print 'example: getting loss'
        
        #target_var = T.matrix('target_var')
        target_var = T.as_tensor_variable(
            np.ones((5,2)).astype(np.float32)    
        )
        weight = 0
        dataset_size = 10000
        weights = get_output(hnet,ep)
        logdets = get_output(ld,ep)
        
        loss, _ = get_elbo(output_var,
                           target_var,
                           weights,
                           logdets,
                           weight,
                           dataset_size,
                           prior=log_normal,
                           lbda=0,
                           output_type = 'categorical')
        print loss.eval()
        
        output_var = N_get_output(layer,input_var,hnet,ep,
                                  static_bias=static_bias,
                                  norm_type='BN',test_time=True)
        output_var.eval()
        
        
        
