import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F


class Control_Variate(nn.Module):
    def __init__(self,input_Dim,hidden_Dim):
        super(Control_Variate,self).__init__()
        self.fc0 = nn.Linear(input_Dim,1)
        self.fc1 = nn.Linear(1,hidden_Dim)
        self.fc2 = nn.Linear(hidden_Dim,1)
        self.Sig = F.sigmoid
        self.Tanh = F.tanh
    def forward(self, x):
        h0 = self.Sig(self.fc0(x))
        h1 = self.Tanh(self.fc1(h0))
        h2 = self.fc2(h1)
        return self.Tanh(h2)

def Adam_Optim(t, m, v, g, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    '''
    Implement Adam Optimizer
    '''

    m_t = beta1*m + (1-beta1)*g
    v_t = beta2*v + (1-beta2)*g*g

    m_cap = m_t/(1-(beta1**t))
    v_cap = v_t/(1-(beta2**t))

    lr = lr * (1 - beta2 ** t) ** (1 / 2) / (1 - beta1 ** t)
    g = (lr*m_cap)/(v_cap**(1/2) + eps)

    return g,m_t,v_t



def prob_mapping(phi):
    '''
    Given phi; apply logistic func to it map to (0,1)
    :return: 
    '''

    theta = torch.exp(phi)/(1+torch.exp(phi))
    theta = theta.clamp(min=1e-4,max=1-1e-4) # play safe
    return theta

def d_log_q_z_given_x(b,phi):
    '''
    Derivative of Bernoulli
    '''
    sig = torch.sigmoid(-phi)
    return b*sig-(1-b)*(1-sig)

def relax_from_discrete(phi,sample_size,dim):
    u = torch.FloatTensor(sample_size, dim).uniform_(0, 1) + 1e-8
    phi = prob_mapping(phi)
    return torch.log(phi/(1-phi)) + torch.log((u) / (1 - u))

def func_in_expect(b, t):
    return torch.mean((b - t)**2, dim=1)

def H_z(z,batch_size,num_latents):
    return torch.where(z>0,torch.ones([batch_size,num_latents]),torch.zeros([batch_size,num_latents]))


def relax_reparam(phi,b,sample_size,dim):
    phi = prob_mapping(phi)
    v = torch.FloatTensor(sample_size, dim).uniform_(0, 1) + 1e-8
    v_prime = b * (v * phi + (1 - phi)) + (1 - b) * v * (1-phi)
    z_tilde = torch.log(phi/(1-phi)) + torch.log((v_prime) / (1 - v_prime))
    return z_tilde

def reBAR_estimator(phi,_lambda,target,sample_size,dim,lr,m,v,t,var_opt):
    z = relax_from_discrete(phi, sample_size, dim)


    h_z = H_z(z, sample_size, dim)
    z_tilde = relax_reparam(phi, h_z, sample_size, dim)
    f_b = func_in_expect(h_z, target)
    f_b = f_b.view([sample_size,1])

    f_z = prob_mapping(z/_lambda)
    f_z_tilde = prob_mapping(z_tilde/_lambda)

    objective_loss = torch.mean(f_b)
    derivative_pdf = d_log_q_z_given_x(h_z, phi)

    d_f_z = torch.autograd.grad(f_z.split(1), phi,create_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(f_z_tilde.split(1), phi,retain_graph=True,create_graph=True)[0]

    reBAR = (f_b - f_z_tilde) * derivative_pdf + (d_f_z - d_f_z_tilde) + 1e-8

    g, m, v = Adam_Optim(t, m, v, reBAR, lr)
    phi.data.sub_(g)
    variance_loss = (reBAR ** 2).mean()

    var_opt.zero_grad()
    variance_loss.backward()
    var_opt.step()


    return phi,objective_loss,variance_loss,m,v,reBAR


def reLAX_estimator(phi,CVnn,target,sample_size,dim,lr,m,v,t,var_opt):

    z = relax_from_discrete(phi, sample_size, dim)
    h_z = H_z(z, sample_size, dim)
    z_tilde = relax_reparam(phi, h_z, sample_size, dim)

    f_b = func_in_expect(h_z, target)
    f_b = f_b.view([sample_size,1])

    f_z = CVnn(z)

    f_z_tilde = CVnn(z_tilde)

    objective_loss = torch.mean(f_b)

    derivative_pdf = d_log_q_z_given_x(h_z, phi)

    d_CV_z = torch.autograd.grad(f_z.split(1), phi,create_graph=True)[0]
    d_CV_z_tilde = torch.autograd.grad(f_z_tilde.split(1), phi,retain_graph=True,create_graph=True)[0]

    g_LAX = (f_b - f_z_tilde) * derivative_pdf + (d_CV_z - d_CV_z_tilde) + 1e-8

    g, m,v = Adam_Optim(t, m, v, g_LAX, lr)
    phi.data.sub_(g)


    variance_loss = (g_LAX ** 2).mean()
    var_opt.zero_grad()
    variance_loss.backward()
    var_opt.step()


    return phi,objective_loss,variance_loss,m,v,g_LAX


def train_Est(target, iters, sample_size, dim, lr):


    phi = torch.zeros(sample_size,dim)
    phi = torch.tensor(phi, requires_grad=True)

    target = torch.ones(sample_size,dim)*target

    # for variance control
    CVnn = Control_Variate(dim,10)
    reLAX_var_opt = torch.optim.Adam(CVnn.parameters(), lr=lr)

    # variance control parameters for reBar

    rebar_phi = torch.zeros(sample_size,dim)
    rebar_phi = torch.tensor(rebar_phi, requires_grad=True)
    _lambda = torch.tensor(5., requires_grad=True)

    reBar_var_opt = torch.optim.Adam([_lambda], lr=lr)


    # momentum, velocity for Adam update
    m=0
    t=0
    v=0

    rebar_m = 0
    rebar_v = 0

    # keep track of loss and variances during the training for reLAX
    variances = []
    obj_loss = []
    theta_curve = []
    lax_curve = []

    # keep track of loss and variances during the training for reBar

    reBar_variances = []
    reBar_obj_loss = []


    for i in range(iters):
        t += 1
        phi, objective_loss, variance_loss, m, v,lax = reLAX_estimator(phi, CVnn, target, sample_size, dim, lr, m, v, t,reLAX_var_opt)
        rebar_phi, rebar_objective_loss, rebar_variance_loss, rebar_m, rebar_v, rebar_lax = reBAR_estimator(rebar_phi, _lambda, target, sample_size, dim, lr,
                                                                                          rebar_m, rebar_v, t, reBar_var_opt)


        theta = torch.mean(prob_mapping(phi)).item()

        rebar_theta = torch.mean(prob_mapping(rebar_phi)).item()
        if i%50==0:
            variances.append(variance_loss.item())
            theta_curve.append(theta)
            loss = theta * (1 - target[0][0]) ** 2 + (1 - theta) * target[0][0] ** 2
            obj_loss.append(loss.item())

            # update reBar estimator's info
            reBar_variances.append(rebar_variance_loss.item())
            rebar_loss = rebar_theta * (1 - target[0][0]) ** 2 + (1 - rebar_theta) * target[0][0] ** 2
            reBar_obj_loss.append(rebar_loss.item())

    return theta_curve,obj_loss,variances,lax_curve, reBar_variances, reBar_obj_loss

if __name__== "__main__":

    #target = 0.55 # both rebar and relax successfully complete the task but reLAX converges much faster
    target = 0.501 # rebar becomes just guessing but reLAX works perfect
    # target = 0.499 # rebar sometime works sometimes fail (based on seed) but reLAX works all times
    torch.manual_seed(4)
    random.seed(4)
    np.random.seed(4)

    theta_curve, obj_loss, variances, lax_curve, reBar_variances, reBar_obj_loss = train_Est(target, 10000, 1, 1, 0.01)
    plt.plot(obj_loss,label="reLAX")
    plt.plot(reBar_obj_loss, label="reBar")
    plt.title('loss with target value={}'.format(target))
    plt.legend()
    plt.show()


    plt.plot(np.log(variances),label="reLAX")
    plt.plot(np.log(reBar_variances), label="reBar")
    plt.title('variance with target value={}'.format(target))
    plt.legend()
    plt.show()
    print(theta_curve[-1])


    # der_shape = torch.ones(f_z.size())
    # f_z.backward(der_shape,retain_graph=True,create_graph=True)
    # d_CV_z = phi.grad.clone()
    # phi.grad.data.zero_()

    # f_z_tilde.backward(der_shape,retain_graph=True,create_graph=True)
    # d_CV_z_tilde = phi.grad.clone()
    # phi.grad.data.zero_()
    # plt.plot(theta_curve,label="theta curve")
    # plt.legend()
    # plt.show()
