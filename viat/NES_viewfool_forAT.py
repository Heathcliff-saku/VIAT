import numpy as np
import os
from evaluate_forAT import comput_fitness
from rendering_image import render_image
from classifier.predict import test_baseline
from tqdm import tqdm
from datasets.opts import get_opts
import joblib
import torch
import time
np.set_printoptions(precision=4,  linewidth=100, suppress=True)
# np.random.seed(0)

def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_normalize(x):
  mean = np.mean(x)
  var = np.var(x)
  y = x-mean/var
  return y

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
  def __init__(self, pi, epsilon=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon
    self.t = 0

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  def _compute_step(self, globalg):
    raise NotImplementedError


class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step


class PEPG:
  '''Extension of PEPG with bells and whistles.'''

  def __init__(self, num_params,  # number of model parameters
               sigma_init=0.10,  # initial standard deviation
               sigma_alpha=0.20,  # learning rate for standard deviation
               sigma_decay=0.999,  # anneal standard deviation
               sigma_limit=0.01,  # stop annealing if less than this
               sigma_max_change=0.2,  # clips adaptive sigma to 20%
               sigma_min=0.05,  # 允许的最小sigma
               sigma_update=True,
               learning_rate=0.01,  # learning rate for standard deviation
               learning_rate_decay=0.9999,  # annealing the learning rate
               learning_rate_limit=0.01,  # stop annealing learning rate
               elite_ratio=0,  # if > 0, then ignore learning_rate
               popsize=256,  # population size
               average_baseline=True,  # set baseline to average of batch
               weight_decay=0.01,  # weight decay coefficient
               rank_fitness=True,  # use rank rather than fitness numbers
               forget_best=True,
               mu_lambda=0.0001,
               sigma_lambda=0.0001):  # don't keep the historical best solution

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_alpha = sigma_alpha
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.sigma_max_change = sigma_max_change
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.average_baseline = average_baseline
    self.sigma_update = sigma_update
    self.sigma_min = sigma_min
    self.mu_lamba = mu_lambda
    self.sigma_lamba = sigma_lambda

    if self.average_baseline:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.batch_size = int(self.popsize / 2)
    else:
      assert (self.popsize & 1), "Population size must be odd"
      self.batch_size = int((self.popsize - 1) / 2)

    # option to use greedy es method to select next mu, rather than using drift param
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.use_elite = False
    if self.elite_popsize > 0:
      self.use_elite = True

    self.forget_best = forget_best
    self.batch_reward = np.zeros(self.batch_size * 2)
    self.mu = np.zeros(self.num_params)
    self.sigma = np.ones(self.num_params) * self.sigma_init
    self.curr_best_mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True  # always forget the best one if we rank
    # choose optimizer
    self.optimizer = SGD(self, learning_rate)

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma * sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
    if self.average_baseline:
      epsilon = self.epsilon_full
    else:
      # first population is mu, then positive epsilon, then negative epsilon
      epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
      self.r = epsilon/self.sigma.reshape(1, self.num_params)
    solutions = self.mu.reshape(1, self.num_params) + epsilon
    self.solutions = solutions
    return solutions

  def comput_entropy(self):
    # 计算每个批次的高斯分布的熵
    r = torch.Tensor(self.r)  # N(0,1)中的采样点
    sigma = torch.Tensor(self.sigma)
    sigma.requires_grad = True
    mu = torch.Tensor(self.mu)
    mu.requires_grad = True
    a = torch.Tensor([60, 180, 60, 1, 0.7, 0.7])    # 各个参数的前系数


    inside = 1-torch.pow(torch.tanh(mu+sigma*r), 2)+1e-8
    neg_logp = torch.log(sigma+1e-8) + 1/2*torch.pow(r, 2) + torch.log(inside)
    entropy = torch.sum(neg_logp, 0)/self.popsize

    Entropy = torch.sum(entropy)

    # 得到的Entropy是当前种群所有参数熵的总和

    # 梯度反向传播
    Entropy.backward()
    # 对于sigma，求这一批次中，每个参数的梯度之和作为这个参数的搜索方向 （11,6）->(1,6)
    # 对于mu，先求出（11,1）的熵，再将每个mu（1,6）对（11,1）的每列求导 得到 （11,6）的梯度
    mu_entropy_grad = mu.grad.clone()
    sigma_entropy_grad = sigma.grad.clone()
    # 梯度清零
    mu.grad.data.zero_()
    sigma.grad.data.zero_()
    self.entropy = Entropy

    return mu_entropy_grad.cpu().detach().numpy(), sigma_entropy_grad.cpu().detach().numpy()





  def tell(self, reward_table_result, mu_entropy_grad, sigma_entropy_grad):
    # input must be a numpy float array
    assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    reward_table = np.array(reward_table_result)

    if self.rank_fitness:
      reward_table = compute_centered_ranks(reward_table)
      # reward_table = compute_normalize(reward_table)

    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay

    reward_offset = 1
    if self.average_baseline:
      b = np.mean(reward_table)
      reward_offset = 0
    else:
      b = reward_table[0]  # baseline

    reward = reward_table[reward_offset:]
    if self.use_elite:
      idx = np.argsort(reward)[::-1][0:self.elite_popsize]
    else:
      idx = np.argsort(reward)[::-1]

    best_reward = reward[idx[0]]
    if (best_reward > b or self.average_baseline):
      best_mu = self.mu + self.epsilon_full[idx[0]]
      best_reward = reward[idx[0]]
    else:
      best_mu = self.mu
      best_reward = b

    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.sigma = np.ones(self.num_params) * self.sigma_init
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # short hand
    epsilon = self.epsilon
    sigma = self.sigma

    # update the mean

    # move mean to the average of the best idx means
    if self.use_elite:
      self.mu += self.epsilon_full[idx].mean(axis=0)
    else:
      rT = (reward[:self.batch_size] - reward[self.batch_size:])
      change_mu = np.dot(rT, epsilon) + self.mu_lamba*mu_entropy_grad
      #print('rt:\n', rT)
      #print('epsilon', epsilon)
      # print('mu-loss1:', np.dot(rT, epsilon))
      # print('mu-loss2:', self.mu_lamba*mu_entropy_grad)

      self.optimizer.stepsize = self.learning_rate
      update_ratio = self.optimizer.update(-change_mu)  # adam, rmsprop, momentum, etc.
      # self.mu += (change_mu * self.learning_rate) # normal SGD method

    # adaptive sigma
    # normalization
    
    #if (self.sigma[a] > self.sigma_min for a in range(self.num_params)):
    if (self.sigma_alpha > 0 and self.sigma_update):
      stdev_reward = 1.0
      if not self.rank_fitness:
        stdev_reward = reward.std()
      S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
      reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
      rS = reward_avg - b

      delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

      # adjust sigma according to the adaptive sigma calculation
      # for stability, don't let sigma move more than 10% of orig value
      change_sigma = self.sigma_alpha * (delta_sigma + self.sigma_lamba*sigma_entropy_grad)

      # print('sigma-loss1:', delta_sigma)
      # print('sigma-loss2:', self.sigma_lamba*sigma_entropy_grad)

      change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
      change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
      self.sigma += change_sigma
      self.sigma = np.clip(self.sigma, 0.0, 0.15)

      if (self.sigma_decay < 1):
        self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

    if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  def current_param(self):
    return self.curr_best_mu

  def set_mu(self, mu):
    self.mu = np.array(mu)

  def best_param(self):
    return self.best_mu

  def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma, self.entropy)


def NES_viewfool_search_step(model, label, ckpt_path, mood, mu_start=None, sigma_start=None):
    args = get_opts()
    MAX_ITERATION = args.iteration
    POPSIZE = args.popsize
    NUM_PARAMS = 6
    N_JOBS = 3
    # 搜索六维空间，th phi gamma r x y
    solver = PEPG(num_params=NUM_PARAMS,  # number of model parameters
                  sigma_init=0.1,  # initial standard deviation
                  sigma_update=True,  # 不大幅更新sigma
                  learning_rate=0.1,  # learning rate for standard deviation
                  learning_rate_decay=0.99,
                  learning_rate_limit=0,  # don't anneal the learning rate
                  popsize=POPSIZE,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=0.01,
                  sigma_lambda=0.01
                  )

    logging = {'mu': [], 'sigma': [], 'fitness': [], 'entropy':[]}
    history = []
    fitness_origin = []
    history_best_solution = []
    for j in range(MAX_ITERATION):
      solutions = solver.ask()
      mu_entropy_grad, sigma_entropy_grad = solver.comput_entropy()

      # gamma (-30,30)
      solutions[:, 0] = 30 * np.tanh(solutions[:, 0])
      # th (-180,180)
      solutions[:, 1] = 180 * np.tanh(solutions[:, 1])
      # phi (-70, 70)
      solutions[:, 2] = 70 * np.tanh(solutions[:, 2])
      # r (3, 5)
      solutions[:, 3] = np.tanh(solutions[:, 3]) + 4
      # x (-0.5, 0.5)
      solutions[:, 4] = 0.5 * np.tanh(solutions[:, 4])
      # x (-0.5, 0.5)
      solutions[:, 5] = 0.5 * np.tanh(solutions[:, 5])

      fitness_list = np.zeros(solver.popsize)


      #  多进程工作
      # with joblib.Parallel(n_jobs=N_JOBS) as parallel:
      #   #for i in tqdm(range(solver.popsize)):
      #     #fitness_list[i] = comput_fitness(solutions[i])

      #   fitness_list = parallel(joblib.delayed(comput_fitness)(solutions[i], solver.sigma) for i in tqdm(range(solver.popsize)))

      fitness_list = comput_fitness(model, label, ckpt_path, solutions, is_viewfool=True)

      solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad)
      result = solver.result()  # first element is the best solution, second element is the best fitness

    mu = result[0]
    sigma = result[3]
    entropy = result[4]
    return mu, sigma, entropy

class ViewFool:
  def __init__(self, mood):
    args = get_opts()
    if mood == 'train':
      all_class = os.listdir(f'{args.ckpt_attack_path}/train/')
    else: # eval_test
      all_class = os.listdir(f'{args.ckpt_attack_path}/test/')
    all_class.sort()
    class_num = len(all_class)
    self.class_num = class_num
    self.all_class = all_class
    # object_num = len(os.listdir(f'{args.ckpt_attack_path}/' + all_class[0] + '/'))
    object_num = 10

    self.dist_pool_mu_return = np.zeros([class_num, object_num, 6])
    self.dist_pool_sigma_return = np.zeros([class_num, object_num, 6])

  def step_viewfool(self, class_id, all_class, model, mood):
      args = get_opts()
      object_num = 10
      mu_result = np.zeros([object_num, 6])
      sigma_result = np.zeros([object_num, 6])
      Entropy_class = 0

      if mood == 'train':
        path = f'{args.ckpt_attack_path}/train/'
      else:
        path = f'{args.ckpt_attack_path}/test/'

      all_object = os.listdir(path + all_class[class_id] + '/')
      all_object.sort()
      
      for object_id in range(len(all_object)):
        ckpt_path = path + all_class[class_id] + '/' + all_object[object_id]
        label = int(all_class[class_id])
        mu, sigma, entropy = NES_viewfool_search_step(model, label, ckpt_path, mood)

        mu_result[object_id, :] = mu
        sigma_result[object_id, :] = sigma
        Entropy_class += entropy
        
      return mu_result, sigma_result, Entropy_class

def NES_viewfool_search(model, mood='test'):

  args = get_opts()
  method = 'para'
  viewfool_solver = ViewFool(mood=mood)

  dist_pool_mu_return = np.zeros([viewfool_solver.class_num, 10, 6])
  dist_pool_sigma_return = np.zeros([viewfool_solver.class_num, 10, 6])
  
#-----------------------------------并行-------------------------------------------------------------
  if method == 'para':
    with joblib.Parallel(n_jobs=5) as parallel:
      res = parallel(joblib.delayed(viewfool_solver.step_viewfool)(class_id, viewfool_solver.all_class, model, mood) for class_id in tqdm(range(viewfool_solver.class_num)))
      mu_result = [item[0] for item in res]
      sigma_result = [item[1] for item in res]
      Entropy = [item[2] for item in res]

      average_entropy = sum(Entropy)/len(Entropy)
    for i in range(len(mu_result)):
      dist_pool_mu_return[i, :, :] = mu_result[i]
      dist_pool_sigma_return[i, :, :] = sigma_result[i]

#-----------------------------------串行-------------------------------------------------------------
  else:
    for class_id in tqdm(range(viewfool_solver.class_num)):
      mu_result, sigma_result = viewfool_solver.step_viewfool(class_id, viewfool_solver.all_class, model, mood)
      dist_pool_mu_return[i, :, :] = mu_result
      dist_pool_sigma_return[i, :, :] = sigma_result

  print("entropy:", average_entropy)
  np.save(f'./dist_pool/dist_pool_mu_{args.AT_exp_name}_viewfool_{mood}.npy', dist_pool_mu_return)
  np.save(f'./dist_pool/dist_pool_sigma_{args.AT_exp_name}_viewfool_{mood}.npy', dist_pool_sigma_return)