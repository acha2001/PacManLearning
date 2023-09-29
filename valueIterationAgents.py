# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*
      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
          mdp.isTerminal(state)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.runValueIteration()
  
  def runValueIteration(self):
    
    states= self.mdp.getStates()
    
    # Set number of iterations
    for i in range(self.iterations): 
      tmp = self.values.copy() # copy of current values
      for s in states:
        actions = self.mdp.getPossibleActions(s)
        values=[]
        
        for a in actions:
          q_vals = self.computeQValueFromValues(s,a)
          values.append(q_vals)
          if not a:
              tmp[s] = 0
          else:
              tmp[s] = max(values)
      self.values=tmp
  
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]
  
  def computeQValueFromValues(self, state, action):
      """
        value function stored in self.values.
      """
      "*** YOUR CODE HERE ***"
      transitions = self.mdp.getTransitionStatesAndProbs(state, action)
      q_val=0
      for new_s, prob in transitions:
          
          value = self.getValue(new_s)
          reward = self.mdp.getReward(state, action, new_s)
          q_val += prob*(reward+self.discount*value)
      return q_val
      util.raiseNotDefined()
  
  def computeActionFromValues(self, state):
    """
      The policy is the best action in the given state
      according to the values currently stored in self.values.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
      return None
    else:
      actions = self.mdp.getPossibleActions(state)
      q_value = [(self.getQValue(state, a),a) for a in actions]
      return max(q_value, key=lambda item:item[0])[1]            
    util.raiseNotDefined()

  def getPolicy(self, state):
    return self.computeActionFromValues(state)

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.computeActionFromValues(state)

  def getQValue(self, state, action):
      return self.computeQValueFromValues(state, action)
  
class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

      states = self.mdp.getStates()
      
      for i in range(self.iterations):
          
        tmp_vals = self.values.copy() # copy of current values
        state = states[i%len(states)] #grab single states
        actions = self.mdp.getPossibleActions(state)     
        values=[]
        
        for a in actions:
            
            
            
          if not a:
            tmp_vals[state]=0
          else:
            q_vals = self.computeQValueFromValues(state, a)
            values.append(q_vals)
            tmp_vals[state] = max(values)
        
        self.values = tmp_vals
          
      
      states = self.mdp.getStates()

    
        
            
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = self.getPredecessors()
        
        states = self.mdp.getStates()

        q = util.PriorityQueue() #initialize an empty priority Queue

        for s in states:
            if not self.mdp.isTerminal(s):
                diff = abs(self.values[s] - self.maxQValue(s)) #get the absolute value diff
                q.update(s, -diff) #push s onto priority Queue with priority -diff, update its priority if needed

        for i in range(self.iterations):   
          if q.isEmpty(): # base case
            return

          state = q.pop()  
          self.values[state] = self.maxQValue(state) 

          for pred in list(predecessors[state]):
            diff = abs(self.values[pred] - self.maxQValue(pred)) #get the absolute value diff
            q.update(pred, -diff) #pred priority update with priority -diff


    def maxQValue(self, state):
        actions = self.mdp.getPossibleActions(state)    
        
        if not actions:
            return None

        Q_vals = []
        for a in actions:
            Q_vals.append(self.computeQValueFromValues(state, a))
            
        return max(Q_vals)


    def getPredecessors(self):
        
        predecessors = {}
        states = self.mdp.getStates()
        
        # create a set for each state
        for s in states: # initalize to sets
          predecessors[s]= set()
        
        for s in states:
           # Find all actions for each state
           actions = self.mdp.getPossibleActions(s)
           
           for a in actions:
              # Find each possible child state from each action
              for childState, prob in self.mdp.getTransitionStatesAndProbs(s,a):  
                predecessors[childState].add(s)

        return predecessors



