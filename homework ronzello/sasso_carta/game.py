import numpy as np
from bandit import bandit
import matplotlib.pyplot as plt
dim_A = 5
Action = ["scissor","paper","rock","lizard","spock"]
T = 400
eps_agent = 0.1
eps_player = 0.9
def epsilon_greedy(Q):
    if np.random.random() < eps_agent:
        a = np.random.randint(0,dim_A)
    else:
        a = np.argmax(Q)
    return a

def stationary_player():
    if np.random.random() < eps_player:
        a = np.random.randint(0,dim_A)
    else:
        a = 2 #rock
    return a
def non_stationary_player(t):
    if np.random.random() < eps_player:
        a = np.random.randint(0,dim_A)
    else:
        a = int(t/80) % 5  #ogni 80 giocate cambia la sua mossa preferita
    return a

def play_eps(sta=True):
    Q = np.zeros((dim_A,T))
    N = np.zeros((dim_A,T))

    alp=0.1
    for t in range(T-1):
        a_agent = epsilon_greedy(Q[:,t])
        a_player = stationary_player() if sta else non_stationary_player(t)
        r = bandit(a_agent,a_player)
        N[:,t+1] = N[:,t]
        N[a_agent,t+1] = N[a_agent,t]+1
        #aggiorna usando media empirica iterativa
        Q[:,t+1] = Q[:,t]
        if sta:
            Q[a_agent,t+1] = Q[a_agent,t] + 1/N[a_agent,t+1]*(r-Q[a_agent,t])
        else:
            Q[a_agent,t+1] = Q[a_agent,t] + alp*(r-Q[a_agent,t+1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Primo subplot — come "nexttile"
    #axes[0].step(range(1, T + 1), N.T, linewidth=2)
    axes[0].set_xlim([1, T])
    axes[0].set_ylabel("N (conteggi azioni)")
    for i, action in enumerate(Action):
        axes[0].step(range(1, T + 1), N[i, :], linewidth=2, label=action)

    axes[0].legend(title="Azione", loc="best")


    # Secondo subplot
    axes[1].step(range(1, T + 1), Q.T, linewidth=2)
    axes[1].set_xlim([1, T])
    axes[1].set_ylabel("Q (stima del valore)")
    axes[1].set_xlabel("Tempo")

    fig.savefig(f"eps_stationary={sta}.png", dpi=300)



def play_ucb(sta=True):
    Q = np.ones((dim_A,T))*1.1
    N = np.ones((dim_A,T))
    c = 0.3
    alp = 0.1
    for t in range(T-1):
        a_agent = np.argmax(np.add(Q[:,t],c*np.sqrt(np.log(t+1)/N[:,t])))
        a_player = stationary_player() if sta else non_stationary_player(t)
        r = bandit(a_agent,a_player)
        N[:,t+1] = N[:,t]
        N[a_agent,t+1] = N[a_agent,t]+1

        Q[:,t+1] = Q[:,t]
        if sta:
            Q[a_agent,t+1] = Q[a_agent,t] + 1/N[a_agent,t+1]*(r-Q[a_agent,t])
        else:
            Q[a_agent,t+1] = Q[a_agent,t] + alp*(r-Q[a_agent,t])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Primo subplot — come "nexttile"
    #axes[0].step(range(1, T + 1), N.T, linewidth=2)
    axes[0].set_xlim([1, T])
    axes[0].set_ylabel("N (conteggi azioni)")
    for i, action in enumerate(Action):
        axes[0].step(range(1, T + 1), N[i, :], linewidth=2, label=action)

    axes[0].legend(title="Azione", loc="best")

    # Secondo subplot
    axes[1].step(range(1, T + 1), Q.T, linewidth=2)
    axes[1].set_xlim([1, T])
    axes[1].set_ylabel("Q (stima del valore)")
    axes[1].set_xlabel("Tempo")

    fig.savefig(f"ucb_stationary={sta}.png", dpi=300)


def preference_update(sta = True):
    H = np.zeros(dim_A)
    historyH = np.zeros((dim_A, T))
    historyProb = np.zeros((dim_A, T))
    historyN = np.zeros((dim_A, T + 1))
    bR=0
    beta=0.1
    alpha=0.1

    for t in range(T):
        expH = np.exp(H - np.max(H))  # stabilità numerica
        Prob = expH / np.sum(expH)


        a = np.searchsorted(np.cumsum(Prob), np.random.rand())
        a_player = stationary_player() if sta else non_stationary_player(t)
        r = bandit(a,a_player)

        # Aggiornamento reward medio
        bR = bR + beta * (r - bR)

        # Aggiornamento preferenze
        H = H - alpha * (r - bR) * Prob
        H[a] = H[a] + alpha * (r - bR) * (1 - Prob[a])


        historyH[:, t] = H
        historyProb[:, t] = Prob
        historyN[a, t] += 1
        historyN[:, t + 1] = historyN[:, t]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(historyH.T, linewidth=2)
    axes[0].set_xlim([0, T])
    axes[0].set_ylabel("H (preferenze)")

    axes[1].plot(historyProb.T, linewidth=2)
    axes[1].set_xlim([0, T])
    axes[1].set_ylabel("Probabilità softmax")

    axes[2].plot(historyN[:, :-1].T, linewidth=2)
    axes[2].set_xlim([0, T])
    axes[2].set_ylabel("Conteggio azioni")
    axes[2].set_xlabel("Tempo")

    fig.savefig(f"pref_stationary={sta}.png", dpi=300)


if __name__ == "__main__":
    np.random.seed(42)
    play_eps()
    play_eps(False)
    play_ucb()
    play_ucb(False)
    preference_update()
    preference_update(False)