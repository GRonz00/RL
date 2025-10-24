from random import randint
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random

from racetrack import create_track,get_start_point
V = 5
A = 9  # The actions are increments to the velocity components. Each may be changed by +1,  1, or 0 in each step, for a total of nine (3 ⇥ 3) actions.


REWARD_FISSO = True
#v2 nello stato ho solo la posizione
def generate_episode(track,pi,eps):
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]

    """
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

    ao, av = actions[randint(0, A-1)]
    #inizia la corsa
    episode = []
    end = False
    for _ in range(1000):  #limito il numero di step, le prime policy casuali potrebbero non toccare la
        stato_attuale = [s1,s2]
        azione_presa = (ao,av)
        if REWARD_FISSO:
            episode.append([stato_attuale,azione_presa,-1])
        new_vv = vv + av if abs(vv + av)<=V else vv  #se supera la velocità consentita rimane la stessa
        new_vo = vo + ao if abs(vo +ao)<=V else vo
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
                a_idx = pi[s1, s2]

                #On-policy first-visit Monte Carlo control
                if random()>1-eps:
                    a_idx = randint(0,A-1)
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
                a_idx = pi[s1, s2]
                #On-policy first-visit Monte Carlo control
                if random()>1-eps:
                    a_idx = randint(0,A-1)
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
        a_idx = pi[s1, s2]
        if random()>1-eps:
            a_idx = randint(0,A-1)
        ao, av = actions[a_idx]
    return episode

def MC(track):
    Q = np.zeros((N1,N2,A))   #(N1,N2) griglia, azioni
    N = np.zeros((N1,N2,A))
    pi = np.zeros((N1, N2), dtype=int)
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]
    for i in range(N1): #random policy
        for j in range(N2):
            if not track[i,j]==-1:  #non ha senso inizializzare la policy fuori dal tracciato
                pi[i,j] = randint(0,A-1)

    # monitoraggio
    avg_returns = []
    avg_lengths = []
    eps_initial = 0.1
    for el in range(5000):#numero episodi
        #print("episodio "+str(el))
        decay_rate = 0.9995
        eps = eps_initial * decay_rate**el

        episode = generate_episode(track,pi,eps)
        G=0
        total_return = 0
        for t, (stato, azione , reward) in enumerate(reversed(episode)):
            s1, s2 = stato
            a_idx = actions.index(azione)
            G = 0.9 * G + reward  # ritorno con gamma=0.9
            total_return += G
            N[s1, s2, a_idx] += 1
            alpha = 1 / N[s1, s2, a_idx]
            Q[s1, s2, a_idx] += alpha * (G - Q[s1, s2, a_idx])
            best_a = np.argmax(Q[s1,s2,:])
            pi[s1, s2] = best_a
        avg_returns.append(total_return)
        avg_lengths.append(len(episode))

        if (el+1) % 1000 == 0:
            print(f"[Episodio {el+1}] "
                  f"Lunghezza media: {np.mean(avg_lengths[-50:]):.1f}, "
                  f"Return medio: {np.mean(avg_returns[-50:]):.2f}")

    return pi, Q, avg_returns, avg_lengths
def graf(pi, Q, avg_returns, avg_lengths):
    V = np.max(Q, axis=2)

    N1, N2, A = Q.shape

    # Creiamo la griglia di coordinate per il grafico
    x = np.arange(N2)
    y = np.arange(N1)
    X, Y = np.meshgrid(x, y)
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]

    # matrice di vettori per visualizzare la direzione della policy
    U = np.zeros_like(pi, dtype=float)  # componente orizzontale
    Vdir = np.zeros_like(pi, dtype=float)  # componente verticale

    for i in range(N1):
        for j in range(N2):
            if track[i, j] != -1:
                ao, av = actions[pi[i, j]]
                U[i, j] = ao
                Vdir[i, j] = -av  # inverti per coerenza con assi immagine

    # Grafico policy come campo vettoriale (freccette)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(track, cmap='gray_r', origin='upper')
    plt.quiver(X, Y, U, Vdir, color='red', scale=20)
    plt.title("Policy ottimale (direzione delle azioni)")
    plt.xlabel("Posizione X")
    plt.ylabel("Posizione Y")

    # Grafico funzione di valore
    plt.subplot(1, 2, 2)
    masked_V = np.ma.masked_where(track == -1, V)
    plt.imshow(masked_V, cmap='viridis', origin='upper')
    plt.colorbar(label='Valore stimato (V)')
    plt.title("Funzione di valore stimata (Monte Carlo)")
    plt.xlabel("Posizione X")
    plt.ylabel("Posizione Y")

    plt.tight_layout()
    plt.savefig("policy-valore.png",dpi=300)

def run_policy(track, pi, max_steps=500, render=True):
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
        a_idx = pi[s1, s2]
        ao, av = actions[a_idx]
        act.append([ao,av])

        new_vv = vv+av if abs(av+vv)<=V else vv
        # movimento verticale
        step = 1 if new_vv >= 0 else -1
        crash = False
        for y in range(s1, s1 + new_vv + step, step):
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

        s1 = s1 + new_vv
        vv = new_vv

        # movimento orizzontale
        new_vo = vo+ao if abs(vo+ao)<=V else vo
        step = 1 if new_vo >= 0 else -1
        for x in range(s2, s2 + new_vo + step, step):
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

        s2 = s2 + new_vo
        vo = new_vo



    if render:
        print(" Episodio interrotto per limite passi.")
    return trajectory, act


import matplotlib.animation as animation

def run_policy_visual(track, pi, max_steps=500, interval=200, Vmax=5, save=False):
    actions = [(ao, av) for ao in [-1, 0, 1] for av in [-1, 0, 1]]

    N1, N2 = track.shape
    s2 = get_start_point(track)
    s1 = 0
    vo = 0
    vv = 0

    trajectory = [[s1, s2, vo, vv]]
    act = [[]]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_title("Simulazione della gara (Monte Carlo Policy)")
    ax.set_xlabel("Asse X")
    ax.set_ylabel("Asse Y")
    ax.set_xlim(-0.5, N2 - 0.5)
    ax.set_ylim(N1 - 0.5, -0.5)
    ax.imshow(track, cmap='gray_r', origin='upper')

    # Marker per la macchina
    car, = ax.plot([s2], [s1], 'ro', markersize=8, label="Auto")
    path, = ax.plot([s2], [s1], 'r--', linewidth=1, alpha=0.5, label="Traiettoria")
    ax.legend(loc="upper right")

    def update(frame):
        nonlocal s1, s2, vo, vv

        if frame >= max_steps:
            return [car, path]


        a_idx = pi[s1, s2]
        ao, av = actions[a_idx]

        new_vv = vv + av if abs(vv + av) <= Vmax else vv
        new_vo = vo + ao if abs(vo + ao) <= Vmax else vo

        crash = False
        # movimento verticale
        step = 1 if new_vv >= 0 else -1
        for y in range(s1, s1 + new_vv + step, step):
            if not (0 <= y < N1) or track[y, s2] == -1:
                crash = True
                break
            if track[y, s2] == 2:
                trajectory.append([y, s2, vo, vv])
                print("🏁 Arrivato al traguardo!")
                ani.event_source.stop()
                return [car, path]

        if crash:
            s2 = get_start_point(track)
            s1 = 0
            vo = vv = 0
        else:
            s1 += new_vv
            vv = new_vv

            # movimento orizzontale
            step = 1 if new_vo >= 0 else -1
            for x in range(s2, s2 + new_vo + step, step):
                if not (0 <= x < N2) or track[s1, x] == -1:
                    crash = True
                    break
                if track[s1, x] == 2:
                    trajectory.append([s1, x, vo, vv])
                    print("🏁 Arrivato al traguardo!")
                    ani.event_source.stop()
                    return [car, path]

            if crash:
                s2 = get_start_point(track)
                s1 = 0
                vo = vv = 0
            else:
                s2 += new_vo
                vo = new_vo

        trajectory.append([s1, s2, vo, vv])

        # aggiorna grafica
        y_data = [p[0] for p in trajectory]
        x_data = [p[1] for p in trajectory]
        car.set_data([s2], [s1])
        path.set_data(x_data, y_data)

        return [car, path]

    ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=interval, blit=True)

    if save:
        ani.save("simulazione_gara.gif", writer="pillow")

    plt.show()

    return trajectory, act



if __name__ == "__main__":
    N1=20
    N2=20
    track = create_track(N1,N2)

    pi, Q, returns, lengths = MC(track)
    graf(pi, Q, returns, lengths)
    plt.figure()


    plt.plot(returns)
    plt.xlabel("Episodi")
    plt.ylabel("Return")
    plt.title("Andamento del return medio")
    plt.savefig("returns.png",dpi=300)
    plt.figure()

    plt.plot(lengths)
    plt.xlabel("Episodi")
    plt.ylabel("Lunghezza episodio")
    plt.title("Andamento lunghezza episodio")
    plt.savefig("lunghezza.png",dpi=300)
    plt.figure()
    for _ in range(5):
        trajectory, act = run_policy_visual(track, pi)


