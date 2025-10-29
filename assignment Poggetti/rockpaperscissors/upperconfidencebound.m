function  a = upperconfidencebound(Q, t, N, c)

    Qmod = Q + c.*sqrt(log(t)./(N+1));
    a = find(Qmod == max(Qmod),1,"first");