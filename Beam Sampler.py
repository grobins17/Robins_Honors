# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:20:01 2020

@author: Owner
"""
import numpy as np 
import text_utils as txt
from collections import Counter 


def beam_sample_forward(x, transition, emission):
    x = txt.encode_string_to_int_list(x)
    forward_s = np.zeros((len(x)+1, 27)) #array with time steps (including 0) and letters
    forward_s[0, :] = np.ones(27)/27
    states = [*range(len(x) + 1)]
    u = np.zeros(len(x)+1)
    u[0] = 0
    for i in range(1, len(x)+1): #iterate over the time steps
        u[i] = np.random.uniform(0, transition[states[i-1], states[i]])
    for i in range(1, len(x)+1): 
        forward_s[i, :]  = emission[:, x[i-1]]  * np.sum(prob_check(forward_s[i-1,:],u[i])) #the probabity distribution of a state at time i (1, T)  #equals the emission probability at time i
    forward_s[len(x), :] = prob_check(forward_s[len(x),:], u[-1])
    for i in range(1, len(x)+1):
        forward_s[i,:] = forward_s[i,:]/np.sum(forward_s[i,:])
    return forward_s


def beam_sample_forward2(x, states, transition, emission):
    #initialize hidden states -> outside of function
    #sample u -> one for each time step 1-T
    x = txt.encode_string_to_int_list(x)
    u = np.zeros(len(x))
    for i in range(1, len(x)+1):
        u[i-1] = np.random.uniform(0, transition[states[i-1],states[i]])
        #print(u[i-1], transition[states[i-1],states[i]])
    #sample p(s| u, y 1:t)
    forward_s = np.zeros((len(x), 27))
    forward_s[0, :] = emission[:, x[0]]
    forward_s[0, :] = forward_s[0,:] / np.sum(forward_s[0,:])
    for i in range(1, len(x)):
        forward_s[i,:] = emission[:, x[i]] * prob_check(forward_s[i-1, :], u[i])
        forward_s[i,:] = forward_s[i,:] / np.sum(forward_s[i,:])
    return forward_s
    



def beam_sample_backward(forward_s, Transition):
    states = np.zeros(forward_s.shape[0], dtype = int)
    q = np.random.multinomial(1, forward_s[-1,:])
    T = np.argmax(q)
    states[-1] = T
    r = np.zeros((27))
    for i in range(forward_s.shape[0]-2, -1, -1):
        r[:] =  forward_s[i,:]* Transition[:,T]
        r = r/ np.sum(r)
        q = np.random.multinomial(1, r)
        T = np.argmax(q)
        states[i] = T
    print(txt.decode_int_list_to_string(states))
    return states

def prob_check(vec, u):
    a = sum(i for i in vec if i > u)
    return a

Transition = np.loadtxt("typing_transition_matrix.csv", delimiter = ",", skiprows = 1)
Emission = np.loadtxt("typing_emission_matrix.csv", delimiter = ",", skiprows = 1)
d = txt.dict_from_file("brit-a-z.txt")
#valid_words = []
#for i in range(10000):
#    forward_s = beam_sample_forward("kezrninh", Transition, Emission)
#    message = beam_sample_backward(forward_s, Transition, "kezrninh")
#    if txt.check_validity(message, d):
#        valid_words.append(message)
#c = Counter(valid_words)
#print("The most common valid word is: ", end = "")
#print(c.most_common(1)[0][0])
states = [0,1,2,3,4,5,6,7,8]
valid_words = []
for i in range(10000):
    u = beam_sample_forward2("kezrninh", states, Transition, Emission)
    states = beam_sample_backward(u, Transition)
    word = txt.decode_int_list_to_string(states)
    if txt.check_validity(word, d):
        valid_words.append(word)
    states = [0, *states]      
c = Counter(valid_words)
print("The most common valid word is: ", end = "")
print(c.most_common(1)[0][0])