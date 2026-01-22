import numpy as np


class QAgent:
    def __init__(self,
                 n_actions,
                 n_agents,
                 state_size,
                 epsilon,
                 epsilon_min,
                 epsilon_decay,
                 ):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.state_shape = (int(state_size/5),5)
        self.new_state_shape = (self.state_shape[0],self.state_shape[1]+n_agents)

        self.eps = [epsilon]*n_agents
        self.eps_decay_rate = epsilon_decay
        self.min_eps = epsilon_min

        #build model
        self.model = self.build_GRU_network()
        self.target_model = self.build_GRU_network()
        self.target_model.set_weights(self.model.get_weights())

    def choose_actions(self,state,id):
        if np.random.uniform()<self.eps[id]:
            action = np.random.randint(0,self.n_actions)
        else:
            #add agent id one-hot form to observation
            inputs = self.add_agent_id(id,state)
            inputs = inputs[np.newaxis,:]
            q_values = self.model.predict(inputs)[0]
            action = np.argmax(q_values)
        self.eps[id] = self.eps[id] * self.eps_decay_rate
        self.eps[id] = max(self.eps[id],self.min_eps)
        return action

    def add_agent_id(self,id,state):
        agent_id = [0.0]*self.n_agents
        agent_id[id]=1.0
        agent_id=np.array(agent_id*self.state_shape[0]).reshape(self.state_shape[0],-1)
        state = state.reshape(self.state_shape)
        state = np.hstack((agent_id,state))
        return state

    def build_GRU_network(self):
        return model


    def _hard_update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

        #软更新自己写