from random import randint
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random

N1=20
N2=20
V = 5
A = 9  # The actions are increments to the velocity components. Each may be changed by +1,  1, or 0 in each step, for a total of nine (3 ⇥ 3) actions.

ep = 0.1 #mettere a zero per togliere policy epsilon greedy (pessimi risultai)
REWARD_FISSO = True
#Lo stato è dato dalla posizione sulla griglia e dalla velocità vertica e orizzontale
def create_track(N1,N2):
    track = np.zeros((N1,N2),dtype=int) #fuori =-1, start=1, tracciato= 0 , fine =2
    s1=randint(0,N2-10)
    s2=randint(s1+5,N2)
    for i in range(N2):  #crea linea di start
        if i in range(s1,s2):
            track[0,i]=1
        else:
            track[0,i]=-1
    y=s1
    for i in range(1,N1):   #lato sinistro del percorso
        x = np.random.choice([-1, 0, 1], p=[0.3, 0.3, 0.4]) #continuita del percorso al massimo si aggiunge o toglie una cella allariga dopo, cambia probabilità per scegliere larghezza strada
        y = y + x if 0 <= y + x <= N2 - 5  else y
        for j in range(y):
            track[i,j]=-1
    y=s2
    for i in range(1,N1-5): #lato destro lascio le ultime 5 che linea di fine
        x = np.random.choice([-1, 0, 1], p=[0.4, 0.3, 0.3]) #continuita del percorso al massimo si aggiunge o toglie una cella allariga dopo
        if 0 <= y + x < N2 and not track[i,y-5]==-1:  ##lascia sempre almento un percorso di 4 celle
            y = y + x
        if track[i,y-5]==-1:
            y+=1
        for j in range(y,N2):
            track[i,j]=-1
        #if track[i,y-2]==-1:
         #   track[i,y]=track[i,y-1]=0
    for i in range(1,6): #fine
        track[N1-i,N2-1]=2
    for riga in track:
        print(riga)
    return track
def get_start_point(track):
    for i in range(N2):
        if track[0,i]==1:
            x=i
            break
    y=N2-1
    for i in range(x,N2):
        if not track[0,i]==1:
            y=i
            break
    return randint(x,y)

# v1 nello stato mantengo sia posizione che velocità
def generate_episode(track,pi):
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]

    #scegli primo stato casuale
    while True:
        s1 = randint(0,N1-1)
        s2 = randint(0,N2-1)
        if track[s1,s2] == 0 or track[s1,s2]==1:
            break

    vo = randint(-V,V)  #velocita orizzontale
    vv = randint(-V,V)  #vel verticale

    """""
    #scegli il primo stato dalle partenza
    s2 = get_start_point(track)
    s1=0
    vv=0
    vo = 0
    """
    while True: #scegli azione casuale
        ao, av = actions[randint(0, A-1)]
        if check_action(vo,vv,ao,av):
            break
    #inizia la corsa
    episode = []
    end = False
    for _ in range(1000):  #limito il numero di step, le prime policy casuali potrebbero non toccare la
        stato_attuale = [s1,s2,vo,vv]
        azione_presa = (ao,av)
        if REWARD_FISSO:
            episode.append([stato_attuale,azione_presa,-1])
        new_vv = vv + av
        new_vo = vo + ao
        step = 1 if new_vv >= 0 else -1
        stop =  False
        for y in range(s1, s1 + new_vv + step, step): #la macchina si muove prima verticalmente, controllo non sbatta
            if not 0<= y <N2 or track[y,s2]==-1: #va fuori dal tracciato
                if not REWARD_FISSO:
                    episode.append([stato_attuale,azione_presa,-10])
                s2 = get_start_point(track)
                s1=0
                vv=0
                vo = 0
                a_idx = pi[s1, s2, vo+V, vv+V]

                #On-policy first-visit Monte Carlo control
                if random()>1-ep:
                    valid_action = [a for a,(ao,av) in enumerate(actions) if check_action(vo, vv, ao, av)]
                    a_idx = valid_action[randint(0,len(valid_action)-1)]

                ao, av = actions[a_idx]
                stop = True
                break
        if stop:
            continue
        #aggiorno stato verticale dopo il controllo
        s1 += new_vv
        vv = new_vv
        #inizio controllo orizzontale
        step = 1 if new_vo >= 0 else -1
        for x in range(s2, s2 + new_vo + step, step):
            if not 0<= x <N1 or track[s1,x]==-1: #va fuori dal tracciato
                if not REWARD_FISSO:
                    episode.append([stato_attuale,azione_presa,-1])
                s2 = get_start_point(track)
                s1=0
                vv = 0
                vo=0
                a_idx = pi[s1, s2, vo+V, vv+V]
                #On-policy first-visit Monte Carlo control
                if random()>1-ep:
                    valid_action = [a for a,(ao,av) in enumerate(actions) if check_action(vo, vv, ao, av)]
                    a_idx = valid_action[randint(0,len(valid_action)-1)]
                ao, av = actions[a_idx]
                stop = True
                break
            if track[s1,x]==2: #arrivata alla fine
                if not REWARD_FISSO:
                    episode.append([stato_attuale,azione_presa,30])
                end=True
                break
        if stop:
            continue
        if end:
            break
        if not REWARD_FISSO:
            episode.append([stato_attuale,azione_presa,-1])
        s2 = s2 + new_vo
        vo = new_vo
        a_idx = pi[s1, s2, vo+V, vv+V]
        if random()>1-ep:
            valid_action = [a for a,(ao,av) in enumerate(actions) if check_action(vo, vv, ao, av)]
            a_idx = valid_action[randint(0,len(valid_action)-1)]
        ao, av = actions[a_idx]
    return episode






def MC_exploring_start(track):
    Q = np.zeros((N1,N2,2*V+1,2*V+1,A))   #(N1,N2) griglia, vel orizzonta e verticale, azioni
    N = np.zeros((N1,N2,2*V+1,2*V+1,A))
    pi = np.zeros((N1, N2, 2*V+1, 2*V+1), dtype=int)
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]
    for i in range(N1): #random policy
        for j in range(N2):
            if not track[i,j]==-1:  #non ha senso inizializzare la policy fuori dal tracciato
                for m in range(2*V+1):
                    for n in range(2*V+1):
                        valid_actions = []
                        for a, (ao, av) in enumerate(actions):
                            if check_action(m-V, n-V, ao, av):
                                valid_actions.append(a)

                        if valid_actions:
                            pi[i, j, m, n] = valid_actions[randint(0, len(valid_actions)-1)]

    # monitoraggio
    avg_returns = []
    avg_lengths = []
    for el in range(100000):#numero episodi
        #print("episodio "+str(el))

        episode = generate_episode(track,pi)
        G=0
        total_return = 0
        for stato, azione , reward in reversed(episode):
            s1, s2, vo, vv = stato
            a_idx = actions.index(azione)
            G = 0.9 * G + reward  # ritorno con gamma=0.9
            total_return += G
            N[s1, s2, vo+V, vv+V, a_idx] += 1
            alpha = 1 / N[s1, s2, vo+V, vv+V, a_idx]
            Q[s1, s2, vo+V, vv+V, a_idx] += alpha * (G - Q[s1, s2, vo+V, vv+V, a_idx])

            valid_actions = [a for a,(ao,av) in enumerate(actions) if check_action(vo, vv, ao, av)]

            best_a = max(valid_actions, key=lambda a: Q[s1, s2, vo+V, vv+V, a])
            pi[s1, s2, vo+V, vv+V] = best_a
        avg_returns.append(total_return)
        avg_lengths.append(len(episode))

        if (el+1) % 1000 == 0:
            print(f"[Episodio {el+1}] "
                  f"Lunghezza media: {np.mean(avg_lengths[-50:]):.1f}, "
                  f"Return medio: {np.mean(avg_returns[-50:]):.2f}")

    return pi, Q, avg_returns, avg_lengths


def check_action(vo,vv,ao,av):
    if abs(vo+ao)<=V and abs(vv+av)<=V and not av+vv+ao+vo==0:
        return True
    return False
def run_policy(track, pi, max_steps=500, render=True):
    """
    Esegue un episodio seguendo la policy appresa.
    Restituisce il percorso seguito (lista di stati).
    """
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]

    # scegli punto di partenza dalla linea di start
    s2 = get_start_point(track)
    s1 = 0  # partenza sempre dall'alto
    vo = 0
    vv = 0

    trajectory = []
    act = [[]]

    for step_count in range(max_steps):
        trajectory.append([s1, s2, vo, vv])
        # azione greedy dalla policy
        a_idx = pi[s1, s2, vo+V, vv+V]
        ao, av = actions[a_idx]
        act.append([ao,av])

        # movimento verticale
        step = 1 if (vv + av) >= 0 else -1
        crash = False
        for y in range(s1, s1 + vv + av + step, step):
            if not (0 <= y < N1) or track[y, s2] == -1:
                crash = True
                break
            if track[y, s2] == 2:
                trajectory.append([y, s2, vo, vv])
                if render: print("Arrivato al traguardo!")
                return trajectory, act

        if crash:
            s2 = get_start_point(track)
            s1=0
            vv = 0
            vo=0
            if render: print(" Crash")
            continue

        s1 = s1 + vv + av
        vv = vv + av

        # movimento orizzontale
        step = 1 if (vo + ao) >= 0 else -1
        for x in range(s2, s2 + vo + ao + step, step):
            if not (0 <= x < N2) or track[s1, x] == -1:
                crash = True
                break
            if track[s1, x] == 2:
                trajectory.append([s1, x, vo, vv])
                if render: print("Arrivato al traguardo!")
                return trajectory, act

        if crash:
            s2 = get_start_point(track)
            s1=0
            vv = 0
            vo=0
            if render: print("Crash")
            continue

        s2 = s2 + vo + ao
        vo = vo + ao



    if render:
        print(" Episodio interrotto per limite passi.")
    return trajectory, act
if __name__ == "__main__":
    track = create_track(N1,N2)
    pi, Q, returns, lengths = MC_exploring_start(track)

    plt.plot(returns)
    plt.xlabel("Episodi")
    plt.ylabel("Return")
    plt.title("Andamento del return medio")
    plt.savefig("return_medio.png",dpi=300)

    plt.plot(lengths)
    plt.xlabel("Episodi")
    plt.ylabel("Lunghezza episodio")
    plt.title("Andamento lunghezza episodio")
    plt.savefig("lunghezza_ep.png",dpi=300)
    traj , act= run_policy(track, pi)
    print("Traiettoria:", traj, act )
