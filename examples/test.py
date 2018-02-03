import gym
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers

# LOCAL CONSTS
EPISODES = 100
SHOW = True
SEED = 3623451898

# ACTION CONSTS
CYCLE_CAP = 3
CHAIN_CAP = 3
ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)

# EMBEDDING CONSTS
EMBEDDING = embeddings.ChainEmbedding(3)

# MODEL CONSTS
M = 64
K = 24
LEN = 50
ADJ_PATH = "/home/camoy/tmp/unos_data_adj.csv"
DET_PATH = "/home/camoy/tmp/unos_data_details.csv"
MODEL = models.DataModel(M, K, ADJ_PATH, DET_PATH, LEN)

# LOGGING CONSTS
PATH = "/home/camoy/tmp/"
EXP = 0
CUSTOM = { "agent" : "greedy" }
LOGGING = loggers.CsvLogger(PATH, EXP, CUSTOM)

# MAIN
def main():
	env = gym.make("kidney-v0")
	env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)
	for i in range(EPISODES):
		obs, done = env.reset(), False
		while not done:
			if SHOW:
				env.render()
			obs, reward, done, _ = env.step(1)
			print(obs)

if __name__ == "__main__":
	main()