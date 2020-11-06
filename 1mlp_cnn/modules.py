"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads = {}
        self.params = {}
        self.params['weight'] = np.random.normal(0, 0.0001, (out_features, in_features))
        self.params['bias'] = np.zeros((1, out_features))

        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # raise NotImplementedError
        self.x = x
        out = np.dot(x, self.params['weight'].T) + self.params['bias']
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = np.dot(dout.T, self.x)
        dx = np.dot(dout, self.params['weight'])
        self.grads['bias'] = np.dot(np.ones((1, self.x.shape[0])), dout)
        # print("dw:{}, db:{} ".format(self.grads['weight'].shape, self.grads['bias'].shape))
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        max_x = np.max(x, axis = 1)
        self.exp_x = np.exp(x - np.expand_dims(max_x, axis = 1))
        self.exp_sum = np.expand_dims(np.sum(self.exp_x, axis = 1), axis=1)
        out = np.divide(self.exp_x, self.exp_sum)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # NOT VERIFIED YET
        T = np.multiply(dout, np.divide(self.exp_x, self.exp_sum ** 2))
        sum_T = np.expand_dims(np.sum(T, axis= 1), axis= 1)
        dx = np.multiply((np.divide(dout, self.exp_sum) - sum_T), self.exp_x)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = -1 * np.sum(np.multiply(y, np.log(x))) / x.shape[0]
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dx = -1 * np.divide(y, x) / x.shape[0]
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # x elements greater or equal to 0
        self.x_geq_zero = np.maximum(x, 0)
        # print(self.x_geq_zero)
        
        # x elements less than zero (we take exp(x))
        self.expx_let_zero = np.minimum(np.exp(x), 1)
        # print(self.expx_let_zero)
        
        out = self.x_geq_zero + (self.expx_let_zero - 1)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = np.multiply(dout, self.expx_let_zero)
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx

if __name__ == '__main__':
  S, M, N =3, 4, 5
  x = np.random.randn(S, M)
  # x = np.array([[1,np.e, np.e**2], [np.e, 1, np.e**2]])
  # y = x
  # print(x)

  # test cross
  # cross = CrossEntropyModule()
  # print(cross.forward(x,y))
  
  # test linear
  # linear = LinearModule(M, N)
  # out = linear.forward(x)
  # print(linear.params["bias"])
  # print("Y.shape ", out.shape)
  # dx = linear.backward(out)
  # print("dx.shape ", dx.shape)
  
  # test ELU
  # print(x)
  # elu = ELUModule()
  # out = elu.forward(x)
  # print("forward: ", out)
  # print("dx: ", elu.backward(out))

  #Test softmax
  softmax = SoftMaxModule()
  out = softmax.forward(x)
  print(out)
  print(softmax.backward(out))

  # cross entropy
  # print(x)
  # y = np.random.randint(0,2,x.shape)
  # print(y)
  # cross = CrossEntropyModule()
  # out = cross.forward(x, y)
  # print(out)
  # print(cross.backward(x,y))
