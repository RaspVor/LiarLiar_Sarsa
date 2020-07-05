#Visualisation d'un episode:
Q = defaultdict(lambda: np.zeros(18))
episodee = launcher_stochastic(Q)


epsilon = 0.00 
counter = 0
round_num = 1000
for i in range(round_num):
    counter += launcher_stochastic(Q, epsilon)[-1][2]
print(counter / round_num) 

for i in range(3):
    print(episodee[i])
    

#Action value prediction    
from collections import defaultdict
import numpy as np
import sys



#Action Values
def mc_prediction_q(num_episodes, gamma=1.0, Q = defaultdict(lambda: np.zeros(18)), N = defaultdict(lambda: np.zeros(18))):
    # initialize empty dictionaries of arrays
    
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        # set the value of epsilon
        epsilon = 1.0/((i_episode/8000)+1)    
        
        # generate an episode
        episode = launcher_stochastic(Q, epsilon)
        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # update the sum of the returns, number of visits, and action-value 
        # function estimates for each state-action pair in the episode
        for i, state in enumerate(states):
            old_Q = Q[state][Zactions[actions[i]]] 
            old_N = N[state][Zactions[actions[i]]]
            Q[state][Zactions[actions[i]]] = old_Q + (sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)/(old_N+1)
            N[state][Zactions[actions[i]]] += 1
            
        
    return Q, N


Q, N = mc_prediction_q(1000000, Q = defaultdict(lambda: np.zeros(18)), N = defaultdict(lambda: np.zeros(18)))
 Q, N = mc_prediction_q(10000, 1.0, Q, N)
len(Q)
 
epsilon = 0.00 
counter = 0
round_num = 1000
for i in range(round_num):
    counter += launcher_stochastic(Q, epsilon)[-1][2]
print(counter / round_num) 

len(Q)
sum(([0.47858339, 0.47858339]))

if ((State in Q) and (np.all(Q[State] ==0) == False))  else RL_call_actions_proba




if (((State in Q) == True) and (np.all(Q[State][-2:] ==0) == False)) else RL_call_actions_proba
                                    


 #Control des probas
    def get_probs_pick_actions(Q_s, epsilon, nS = 16, nA=18):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nS
        best_a = np.argmax(Q_s[:-2])
        policy_s[best_a] = 1 - epsilon + (epsilon / nS)
        
        selector = np.array([1,1,1,1,
                             1,1,1,1,
                             1,1,1,1,
                             1,1,1,1,
                             0,0])
        policy_s = np.where(selector==0, 0, policy_s)
        
        return policy_s


    def get_probs_call_actions(Q_s, epsilon, nS = 2, nA=18):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nS
        best_a = np.argmax(Q_s[-2:])+16
        policy_s[best_a] = 1 - epsilon + (epsilon / nS)
        
        selector = np.array([0,0,0,0,
                             0,0,0,0,
                             0,0,0,0,
                             0,0,0,0,
                             1,1])
        policy_s = np.where(selector==0, 0, policy_s)
        
        return policy_s

sum(policy_s)

test = np.array([    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
                        0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
                    -10040.,     0.])

test[:-2]
np.argmax(test[:-2])+15
sum(get_probs_call_actions(test,epsilon))
    
    
    
epsilon = 1.0/((100/8000)+1)    
sum(get_probs_call_actions(Q["1[3 5 1 4 5 2 4 2][0 0 0 0 0 0 0 0][0 0 1 0 0 0 0 0]"],epsilon))
sum(get_probs_pick_actions(Q["0[3 5 2 4 5 2 4 2][0 0 0 0 0 0 0 0][0 0 0 0 0 0 0 0]"],epsilon))


policy_s[0] = 1 - epsilon + (epsilon / nS)










np.argmax(Q["1[1 0 3 0 0 2 4 2][0 2 2 3 3 3 1 0][2 2 2 2 3 2 1 1]"])


V_sorted = sorted(V.items(), key=lambda item:  (item[1][0]), reverse=True)
V_sorted

N_sorted = sorted(N.items(), key=lambda item:  (item[1][0]), reverse=True)

len(Q)
np.argmax(V["0[3 4 5 2 3 3 3 4][0 0 0 0 1 0 0 0][0 0 0 0 0 0 0 0]"])


def get_probs_pick_actions(Q_s, epsilon, nS = 16, nA=18):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nS
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nS)
    
    selector = np.array([1,1,1,1,
                         1,1,1,1,
                         1,1,1,1,
                         1,1,1,1,
                         0,0])
    policy_s = np.where(selector==0, 0, policy_s)
    
    return policy_s


def get_probs_call_actions(Q_s, epsilon, nS = 2, nA=18):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nS
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nS)
    
    selector = np.array([0,0,0,0,
                         0,0,0,0,
                         0,0,0,0,
                         0,0,0,0,
                         1,1])
    policy_s = np.where(selector==0, 0, policy_s)
    
    return policy_s

get_probs_call_actions(Q["1[1 0 3 0 0 2 4 2][0 2 2 3 3 3 1 0][2 2 2 2 3 2 1 1]"],epsilon)

state = "1[1 0 3 0 0 2 4 2][0 2 2 3 0 0 1 0][2 2 2 2 3 2 1 1]"
probabilite = get_probs_call_actions(Q[state], epsilon, nA) \
                                    if state in Q else RL_call_actions_proba

nS=18
nA=16

i_episode = 1
epsilon = 1.0/((80000/8000)+1)
epsilon
policy_s = np.ones(nA) * epsilon / nS
policy_s
1 - epsilon + (epsilon / nS)


test = get_probs(Q["1[1 0 3 0 0 2 4 2][0 2 2 3 3 3 1 0][2 2 2 2 3 2 1 1]"],epsilon, nA)
test
RL_pick_actions_proba = np.array([1/16,1/16,1/16,1/16,
                                      1/16,1/16,1/16,1/16,
                                      1/16,1/16,1/16,1/16,
                                      1/16,1/16,1/16,1/16,
                                      0,0])
    
on prend les éléments à 0 et on les répartit sur les autres    
     
x=np.random.randint(100, size=(1,18))[0]
np.where(RL_pick_actions_proba==0, 0, x)

action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()





returns_sum = defaultdict(lambda: np.zeros(18))
N = defaultdict(lambda: np.zeros(18))
Q = defaultdict(lambda: np.zeros(18))
# loop over episodes

         gamma = 1   
        # generate an episode
        episode = launcher_stochastic()
        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # update the sum of the returns, number of visits, and action-value 
        # function estimates for each state-action pair in the episode
        for i, state in enumerate(states):
            i=0
            returns_sum[states[i]][Zactions[actions[i]]] += sum(rewards[i:]*discounts[:-(1+i)])
            print(returns_sum[state][0])
         
            #N[state][actions[i]] += 1.0
            #Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
            


s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
d = defaultdict(set)
for k, v in s:
    d[k].add(v)

d['red']

d.items()














def mc_prediction_v(num_episodes, gamma=1.0):
    # initialize empty dictionary of lists
    returns = defaultdict(list)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        ## TODO: complete the function
        # generate an episode
        episode = launcher()
        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        # calculate and store the return for each visit in the episode
        for i, state in enumerate(states):
            returns[state].append(sum(rewards[i:]*discounts[:-(1+i)]))
            
    # calculate the state-value function estimate
    V = {k: np.mean(v) for k, v in returns.items()}
        
    return V

V = mc_prediction_v(60000)

V_sorted = sorted(V.items(), key=lambda item:  (item[1]))
#V_sorted

len(V_sorted)


#Action value prediction 

