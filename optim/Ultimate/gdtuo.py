import torch
import torch.nn.functional as F
import math
from classification.optim.Ultimate.mask_get import get_gradient_mask

# STEFunction.apply(input)
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



class Optimizable:

    def __init__(self, parameters, optimizer):
        self.parameters = parameters # a dict mapping names to tensors
        self.optimizer = optimizer   # which must itself be Optimizable!
        self.all_params_with_gradients = []
        self.p_groups = {'masks':[]}
        self.state = {} # optim hp state
        self.lr = 0.
        self.eps = 0.
        
    def initialize(self):
        ''' Initialize parameters, e.g. with a Kaiming initializer. '''
        pass
    
    def begin(self):
        ''' Enable gradient tracking on current parameters. '''
        for param in self.all_params_with_gradients:
             param.grad = None
        self.all_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_() # keep gradient information...
            param.retain_grad()    # even if not a leaf...
            self.all_params_with_gradients.append(param) # 用于添加训练参数
        self.optimizer.begin()

    def zero_grad(self):
        ''' Set all gradients to zero. '''
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    def step(self):
        ''' Update parameters '''
        pass



class NoOpOptimizer(Optimizable):
    '''
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    '''
    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    def step(self, params):
        pass

    def __str__(self):
        return ''

class SGD(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''
    def __init__(self, alpha=0.0001, alpha_eps=0.0001, mu=0.9, config=None, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        self.config = config
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu),
            'alpha_eps':  torch.tensor(alpha_eps),
        }
        super().__init__(parameters, optimizer)
        
        
    def step(self, params):
        self.optimizer.step(self.parameters)        
        for name, param in params.items():
        
            # update hyperparameters other than eps via Gradient Descent
            if name != 'eps' :
                g = param.grad.detach()
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf * self.parameters['mu'] + g
                    g = self.state[name] = buf         
                params[name] = p - g * self.parameters['alpha']
                
            # update eps via Gradient Descent
            elif name == 'eps' and self.config.epsGrad:
                g = param.grad.detach()
                p = param.detach()
                if self.mu != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf * self.parameters['mu'] + g
                    g = self.state[name] = buf
                params[name] = p - g * self.parameters['alpha_eps']
                    
            # if we disable gradient of eps (use fixed eps)             
            else:
                pass
            
    
    def __str__(self):
        return 'sgd / '+ str(self.optimizer)



class MaskedSGD(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''
    def __init__(self, alpha=0.1, mu=0.9, eps=0.05, weight_decay=0.001, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.weight_decay = weight_decay
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu),
            'eps': torch.tensor(eps),
        }
        self.velocity = {}
        super().__init__(parameters, optimizer)
        
    def param_groups(self, PG):
        """ Put mask into optimizer """
        self.p_groups['masks'] = PG['masks'] 
    
    def set_velocity(self, velocity):
        """ Put velocity into optimizer """
        self.velocity = velocity

    def step(self, params):
        
        # Optimize hyperparameters
        self.optimizer.step(self.parameters) 
        
        # obtain lr and eps explicitly
        self.lr = float(self.parameters['alpha'])      
        self.eps = float(self.parameters['eps'])
    
        # Init dict of grads and mask
        grads = {}
        grad_mask = {}
        
        # Get mask via updated eps
        for key in self.velocity.keys(): 
            get_gradient_mask(key, self.velocity[key], grad_mask, self.eps)
        masks = grad_mask
        
        self.sgd(
                params,
                grads,
                masks=masks,
                )
        
        
    def sgd(self, params, grads, masks):
        
        params_with_grad = {}
        for name, param in params.items():
            if param.grad is not None:
                # Get all parameters with gradient
                params_with_grad[name] = param 
                # Detach gradient from computaional graph
                grads[name] = param.grad.detach()
                
        for name, param in params_with_grad.items():
            root_name = name.replace(".weight", "").replace(".bias", "")
            
            grad = grads[name]
            if self.weight_decay != 0:
                grad = grad + param.detach()*self.weight_decay
            
            # Set the gradient of all neurons in the mask to 0 
            if root_name in masks:
                grad[masks[root_name]] = 0.
                
            if self.mu != 0.0:
                if name not in self.state:
                    buf = self.state[name] = grad
                else:
                    buf = self.state[name].detach()
                    buf = buf * self.parameters['mu'] + grad
                grad = self.state[name] = buf
            if root_name in masks:
                grad[masks[root_name]] = 0.
            
            # Get the velocity corresponding to the neuron and update parameters
            if root_name in self.velocity:
                
                if 'bn' in name and 'bias' not in name:
                    velocity_expanded = self.velocity[root_name]
                    params[name] = param.detach() - grad * STEFunction.apply((torch.abs(velocity_expanded) - self.parameters['eps'])) * self.parameters['alpha']
                    
                elif 'conv' in name and 'bias' not in name:
                    extra_dim = len(list(grad.size()))-len(list(self.velocity[root_name].size()))
                    velocity_expanded = self.velocity[root_name].unsqueeze(-1)
                    for _ in range(extra_dim - 1):
                        velocity_expanded = velocity_expanded.unsqueeze(-1)    
                    velocity_expanded = velocity_expanded.repeat(1, list(grad.size())[1], list(grad.size())[2], list(grad.size())[3])
                    params[name] = param.detach() - grad * STEFunction.apply((torch.abs(velocity_expanded) - self.parameters['eps'])) * self.parameters['alpha']
                    
                else:
                    params[name] = param.detach() - grad * self.parameters['alpha']
            
            else:
                if 'bn' in name or 'conv' in name and 'bias' not in name:
                    params[name] = param.detach() - grad * STEFunction.apply((0 - self.parameters['eps'])) * self.parameters['alpha']
                else:
                    params[name] = param.detach() - grad * self.parameters['alpha']


    def __str__(self):
        return 'sgd / '+ str(self.optimizer)



class ModuleWrapper(Optimizable):
    '''
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    '''
    def __init__(self, module, optimizer=NoOpOptimizer()):
        self.module = module
        parameters = {k:v for k, v in module.named_parameters(recurse=True)}
        super().__init__(parameters, optimizer)

    
    def initialize(self):
        self.optimizer.initialize()
    
    def zero_grad(self):
        """ Set all gradients to zero. """
        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()
    
    def forward(self, *xyz):
        return self.module(*xyz)
    
    def train(self):
        self.module.train()
    
    def eval(self):
        self.module.eval()
    
    def step(self):
        self.optimizer.step(self.parameters)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]

            m._parameters[k] = None
            m._parameters[k] = self.parameters[kk]

        for k, v in self.module.named_parameters(recurse=True):
            set_param(self.module, k, v)
