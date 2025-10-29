clear all
close all
clc

% import data
load gambler_MDP.mat

% fattore di sconto per ricompense future 
gamma = 0.9;

% Ottengo le dimensioni del problema 
S = size(P,1);
A = size(R,2);

% inizializzo casualmente la policy
pi = randi(A,S);
% inizializzo casualmente la funzione valore degli stati 
vpi = randn(S,1);

% stati terminali inizializzo a 0.
vpi(1,1) = 0;
vpi(S,1) = 0;

% tengo traccia delle iterazioni
iters = 0;
while true
    % incremento l'iterazione
    iters = iters + 1;

    % funzione per visualizzare la policy e la funzione valore 
    visualize_policy_value(vpi, pi)
    
    vprev = vpi;
    % data la policy la valuto calcolandomi la funzione valore 
    vpi = iterative_policy_evaluation(pi, P, R, gamma, vpi);
    
    % valuto la differenza della nuova funzione valore con quella
    % precedente
    disp(min(vpi-vprev))
    
    % miglioro la policy 
    pip = policy_improvement(vpi, P, R, gamma);

    % finchè la policy non si stabilizza continuo nel loop
    if norm(pip-pi, "inf") == 0
        break;
    else
        pi = pip;
    end
end

% save the value function
save vs_pi.mat vpi pi