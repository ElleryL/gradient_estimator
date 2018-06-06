from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

def prob_map(phi):
    theta = tf.exp(phi)/(1+tf.exp(phi))
    theta = tf.clip_by_value(theta, 1e-5, 1.0-1e-5)
    return theta

def d_log_q_z_given_x(b, phi):

    sig = tf.sigmoid(-phi)
    return b * sig - (1 - b) * (1 - sig)


def relax_reparam(phi, b, batch_size, num_latents):
    phi = prob_map(phi)
    v = tf.random_uniform([batch_size, num_latents], dtype=tf.float32) + 1e-8
    v_prime = b * (v * phi + (1 - phi)) + (1 - b) * v * (1 - phi)
    z_tilde = tf.log(phi/(1-phi)) + tf.log((v_prime) / (1 - v_prime))
    return z_tilde


def relax_from_discrete(phi, batch_size, num_latents):
    u = tf.random_uniform([batch_size, num_latents], dtype=tf.float32) + 1e-8
    phi = prob_map(phi)
    return tf.log(phi/(1-phi)) + tf.log((u) / (1 - u))


def CVnn(z):
    h0 = tf.layers.dense(z,1,tf.nn.sigmoid,name="q_0",use_bias=True)
    h1 = tf.layers.dense(h0, 10, tf.nn.tanh, name="q_1", use_bias=True)
    out = tf.layers.dense(h1, 1,tf.nn.tanh, name="q_out", use_bias=True)
    return out


def func_in_expect(b, t):
    return tf.reduce_mean(tf.square(b - t), axis=1)

def H_z(z,batch_size,num_latents):
    return tf.where(z>0,tf.ones([batch_size,num_latents]),tf.zeros([batch_size,num_latents]))


def reLAX_estimator(phi, target, sample_size, dim, lr):

    # reparameterization variables
    z = relax_from_discrete(phi, sample_size, dim)  # z(u)
    h_z = H_z(z, sample_size, dim)
    z_tilde = relax_reparam(phi, h_z, sample_size, dim)

    # loss function evaluations
    f_b = func_in_expect(h_z, target)

    with tf.variable_scope("CVnn"):
        f_z = CVnn(z)[:, 0]
    with tf.variable_scope("CVnn", reuse=True):
        f_z_tilde = CVnn(z_tilde)

    # loss function for generative model
    objective_loss = tf.reduce_mean(f_b)

    # rebar construction

    d_CV_z = tf.gradients(f_z, phi)[0]

    d_CV_z_tilde = tf.gradients(f_z_tilde, phi)[0]
    derivative_pdf = d_log_q_z_given_x(h_z, phi)

    f_b = tf.reshape(f_b, [sample_size, 1])


    g_LAX = (f_b - f_z_tilde) * derivative_pdf + (d_CV_z - d_CV_z_tilde)

    # variance reduction objective
    variance_loss = tf.reduce_mean(tf.square(g_LAX))

    g_LAX = g_LAX / sample_size

    # optimizers
    grad_opt = tf.train.AdamOptimizer(lr)
    grad_train_op = grad_opt.apply_gradients([(g_LAX, phi)])
    #grad_train_op = phi.assign(phi - lr * g_LAX)

    var_opt = tf.train.AdamOptimizer(lr)
    var_vars = [v for v in tf.trainable_variables() if "CVnn" in v.name]
    var_gradvars = var_opt.compute_gradients(variance_loss, var_list=var_vars)
    var_train_op = var_opt.apply_gradients(var_gradvars)

    with tf.control_dependencies([grad_train_op, var_train_op]):
        train_op = tf.no_op()

    return train_op,variance_loss,objective_loss,g_LAX


def train_RELAX(t,iters ,batch_size, num_latents,lr):
    with tf.Session() as sess:

        phi = tf.Variable([[0.0 for i in range(num_latents)]]*batch_size,trainable=True,)
        target = np.array([[t for i in range(num_latents)]], dtype=np.float32)


        train_op, variance_loss, objective_loss,g_LAX = reLAX_estimator(phi,t,batch_size,num_latents,lr)

        theta = prob_map(phi)

        sess.run(tf.global_variables_initializer())

        variances = []
        obj_loss = []
        theta_curve = []
        lax_curve = []

        for i in tqdm(range(iters)):
            sess.run([train_op])
            _g_LAX = sess.run([g_LAX])
            lax_curve.append(_g_LAX[0][0])
            if i % 50 == 0:
                loss_value,theta_value,var = sess.run([objective_loss, theta,variance_loss])
                tv = tf.reduce_mean(theta_value[0][0])

                theta_curve.append(tv.eval())

                #print(np.log(theta_curve[-1]/(1-theta_curve[-1])))
                loss = tv * (1 - target[0][0]) ** 2 + (1 - tv) * target[0][0] ** 2
                obj_loss.append(loss.eval())
                variances.append(var)
                #print("{}th, loss={}, param={}".format(i, loss_value, theta_curve[-1]))

        return tv, theta_curve, obj_loss, variances,lax_curve

if __name__ == "__main__":
    t = 0.1
    torch.manual_seed(12)
    random.seed(12)
    np.random.seed(12)
    tv, theta_curve, losses, variances,lax_curve = train_RELAX(t, 10000,1, 1,0.01)
    plt.plot(theta_curve)
    plt.show()
    plt.plot(losses)
    plt.show()
    plt.plot(np.log(variances))
    plt.show()
    # plt.plot(lax_curve)
    # plt.show()
    print(theta_curve[-1])