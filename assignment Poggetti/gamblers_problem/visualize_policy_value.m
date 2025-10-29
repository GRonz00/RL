function visualize_policy_value(vpi, pi)

% get the number of states 
S = size(vpi,1);
% Supponiamo che:
% - 'goal' sia l'obiettivo, ad esempio 100
% - V sia il vettore dei valori calcolati per gli stati da 0 a goal (V(1) corrisponde a s=0)
% - policy sia un vettore che contiene le azioni ottimali per gli stati da 1 a goal-1
%   (ricorda che per gli stati terminali (0 e goal) la politica non è definita).

% Definisci il vettore degli stati per il plotting
states = 0:S-1;  

%% Plot della Funzione Valore
figure(1);
plot(states', vpi, '-o', 'LineWidth', 2);
xlabel('Stato (Capitale)');
ylabel('Valore V(s)');
title('Funzione Valore Ottimale');
grid on;

%% Plot della Politica Ottimale
% La politica è definita per gli stati da 1 a goal-1
% (in V, l'indice 1 corrisponde a s=0, perciò policy(2:end-0) corrisponde a s=1,...,goal-1)
figure(2);
plot(1:S-1, pi(2:S), '-o', 'LineWidth', 2);
xlabel('Stato (Capitale)');
ylabel('Importo Scommesso');
title('Politica Ottimale');
grid on;

%% 
figure(3)
subplot(1,2,1);
plot(0:100, vpi, 'LineWidth', 2)
xlabel('Capitale (s)')
ylabel('Probabilità di vincere v*(s)')
title('Funzione valore ottimale')
grid on
%% 

subplot(1,2,2);
plot(0:100, pi, 'LineWidth', 2)
xlabel('Capitale (s)')
ylabel('Scommessa ottimale a = π*(s)')
title('Policy ottimale')
grid on
