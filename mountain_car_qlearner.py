# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:35:15 2019

@author: Acer
"""

import gym
import numpy as np

## Clase Q-learner
# __init__ (self,enviroment) 
# discretizer(self,obs) # Esto no es necesario en todos los casos
# solo cuando tenemos espacios de estados contuniuos, lo mejor es discretizar
# nuestros estados. Para este caso, podemos discretizar tomando intervalos de la 
# siguiente manera [-2,-1],(-1,0],(0,1],(1,2]
# Ahora debemos definir mas hiperparametros de la clase
# 1. Epsilon minimo : El valor más pequeño que se utiliza para llevar el aprendizaje
# es decir , vamos aprendiendo mientras el incremento de aprendizaje sea superior 
# a dicho valor, si no no se toma en cuenta. Esto garantiza la convergencia del algoritmo.
# 2. MAX_NUM_EPISODES: número máximo de iteraciones que estamos dispuestos a realizar.
# 3. STEPS_PER_EPISODE: Número máximo de pasos a realizar a cada episodio.
# 4. ALPHA: Ratio de Aprendizaje del Agente.
# 5. GAMMA: Factor de descuento del agente. -> lo que vamos perdiendo de un paso al otro. Para incentivar
# que el agente llegue lo más pronto a su objetivo. 
# 6. NUM_DISCRETE_BINS: Número de divisiones en el caso de dsicretizar el espacio de estados continuo

# learn(self,obs,action, reward,next_obs)

MAX_NUM_EPISODES = 40000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN /max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

  
class QLearner(object):
    
    def __init__(self,enviroment):
        '''
        Toma el entorno que queremos entrenar . 
        Tomara una instancia del entorno como argumento, y que usaremos para inicializar
        las dimensiones, el tamaño del espacio de observaciones y acciones, determinar
        los parametros para discretizar la observación basandose en el número de secciones(divisiones de
        las observaciones). Se guardara toda esa info en self.              
        '''
        
        # Observaciones
        
        self.obs_shape = enviroment.observation_space.shape # Tamaño ( n filas y n columnas) del espacio de dimensiones
        self.obs_high = enviroment.observation_space.high #Valor más alto del espacio de observaciones
        self.obs_low = enviroment.observation_space.low #Valor más bajo del espacio de observaciones
        self.obs_bins = NUM_DISCRETE_BINS # Número de divisiones para un espacio continuo
        self.bin_width = (self.obs_high - self.obs_low )/self.obs_bins # Ancho de cada espacio discreto
        
        # Acciones
        
        self.action_shape =enviroment.action_space.n # Cuantas acciones puede llevar a cabo
        self.Q = np.zeros((self.obs_bins+1,self.obs_bins +1 , self.action_shape)) # Matriz para guardar los aprendizajes de cada estado
        # En esta matriz guardaremos los estados por los que va pasando el agente.
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0 
        
        
    def discretize(self,obs):
        
        # Nos indica en que intervalo se encuentra la observación
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int)) 
    
    
    def get_action(self,obs):
        
        # Generar una acción a partir de una observación concreta en la que estamos situados
        # Una de las técnicas mas usadas es la de eligir una politica de fuerza bruta basada
        # en minimizar el epsilon. Tomaremos la mejor acción posible que va a estimar el agente 
        # con la mayor probabilidad (1-epsilon) --- epsilon es la probabilidad de equivocarnos
        discrete_obs = self.discretize(obs)
        # Selección de la acción en base a epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:#Con probabilidad 1-e elegimos la mejor posible.
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)]) # Con probabilidad epsilon , elegimos una al azar
        
    def learn(self, obs, action,  reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        
        # Calculamos el aprendizaje
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target - self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.alpha * td_error
        



# Metodo para entrenar el agente


def train(agent,enviroment):
    best_reward = -float('inf') # Es decir nada
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = enviroment.reset() # Primera observación, inicia en 0
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = enviroment.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episodio # {} con recompensa {}, mejor recompensa obtenida {}, epsilon = {}".format(episode,total_reward,best_reward,agent.epsilon))

    ## De todas las politicas de entrenamiento que hemos obtenido, devolvemos la mejor de todas
    
    return np.argmax(agent.Q,axis = 2)

        

# Metodo para ver como de bien le fue al agente:
    
def test(agent,enviroment,policy):
    done  = False
    obs = enviroment.reset()
    total_reward = 0
    while not done:
        action = policy[agent.discretize(obs)] #acción que dictamina la política que hemos entrenado
        next_obs, reward, done, info = enviroment.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward

if __name__ == '__main__':
    
    enviroment = gym.make('MountainCar-v0')
    agent = QLearner(enviroment)
    learned_policy = train(agent,enviroment)
    monitor_path = "./monitor_output"
    enviroment = gym.wrappers.Monitor(enviroment,monitor_path,force = True)
    
    for _ in range(1000):
        test(agent,enviroment,learned_policy)
    enviroment.close()
    


        


