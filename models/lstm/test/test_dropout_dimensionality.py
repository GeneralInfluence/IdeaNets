
import theano
from theano import tensor, config
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def numpy_floatX(data):
        return np.asarray(data, dtype=config.floatX)

# def dropout_layer(self, state_before, use_noise, trng):
#
#     # I need this variable to be for one dimension and repeated for the others.
#
#     temp = state_before[:,:,None]
#
#     binomial2D = trng.binomial(state_before.shape[],
#                     p=0.5, n=1,
#                     dtype=state_before.dtype)
#     tensor.repeat( binomial2D,repeats,axis)
#
#     proj = tensor.switch(use_noise,
#                          (state_before *
#                           trng.binomial(state_before.shape,
#                                         p=0.5, n=1,
#                                         dtype=state_before.dtype)),
#                          state_before * 0.5)
#     return proj
#
#
# tensor.alloc(numpy_floatX(0.),n_samples,dim_proj)
#
#
# proj = (proj * mask[:, :, None]).sum(axis=0)
# proj = proj / mask.sum(axis=0)[:, None]


# Create 2x2 binomial matrix
trng = RandomStreams(1234)
bi_mat = trng.binomial((2,2),p=0.5,n=1,dtype=config.floatX)
f_bi_mat = theano.function([],bi_mat)

bi_rep = np.repeat([f_bi_mat()],3,0)
reshape
bi_shared = theano.shared(bi_rep)
bi_shared.eval()

# # Repeat binomial matrix function
# bi_rep_mat = tensor.repeat(bi_mat,3,0)
# f_bi_mat_rep = theano.function([],bi_rep_mat)
#
# # Repeat same binomial matrix
# bi_mat_rep = bi_mat.repeat(3,0)
# f_bi_mat_rep = theano.function([],bi_mat_rep)
