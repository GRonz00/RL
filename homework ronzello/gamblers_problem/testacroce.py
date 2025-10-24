import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

goal = 100  #il gioco si conclude se vinco 100 dollari o perdo tutto
theta = 1e-9
gamma = 0.9
#Value iteration in place
V=np.zeros(goal + 1)
R=np.zeros(goal + 1)
R[0]=-1
R[goal]=1
p = 0.5 #prob testa
#value iteration algorithn
while True:
    delta = 0
    for s in range(1,goal):
        v = V[s]
        A = np.zeros(min(s,goal-s))#ha senso puntare solo fino all'obiettivo
        for a in range(1,min(s,goal-s)+1):
            A[a-1]=p*(R[s+a]+gamma*V[s+a])+(1-p)*(R[max(0,s-a)]+gamma*V[max(0,s-a)]) #vincita + perdita
        V[s]=max(A)
        delta=max(delta,abs(v-V[s]))
    if delta<theta:
        break

pi = np.zeros(goal + 1, dtype=int)

for s in range(1, goal):
    x = -np.inf
    max_a = 0
    for a in range(1, min(s, goal - s) + 1):
        y = 0.5 * (R[s + a] + gamma * V[s + a]) + \
            0.5 * (R[max(0, s - a)] + gamma * V[max(0, s - a)])
        if y > x:
            x = y
            max_a = a
    pi[s] = max_a

print(pi)
V[0] = R[0]
V[goal] = R[goal]
plt.figure(figsize=(10,4))
plt.plot(V, linewidth=2)
plt.title(f"Value function")
plt.xlabel("Stato s")
plt.ylabel("V(s)")
plt.grid(True)
plt.savefig(f"gambler_V.png", dpi=300)

plt.figure(figsize=(10,4))
plt.bar(np.arange(goal+1),pi)
plt.title(f"Policy ottima")
plt.xlabel("Stato")
plt.ylabel("Puntata")
plt.savefig(f"gambler_policy.png", dpi=300)
