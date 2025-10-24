def bandit(a_agent, a_player):
    win_map = {
        0: [1, 3],
        1: [2, 4],
        2: [0, 3],
        3: [1, 4],
        4: [0, 2],
    }
    if a_agent == a_player:
        return 0
    elif a_player in win_map[a_agent]:
        return 1
    else:
        return -1
