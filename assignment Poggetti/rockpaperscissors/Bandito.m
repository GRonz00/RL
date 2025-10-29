clear all
clc
close all

% EPSILON GREEDY 
% 1 = Sasso, 2 = Carta, 3 = Forbice, 4 = Spock, 5 = Lizard

rng(123);

A = 5; % numero delle azioni possibili
epsilon = 0.2; % probabilità con la quale scelgo un'azione casuale non greedy

% Inizializzo il vettore delle stime delle azioni
Q = zeros(A,1);

% Inizializzo il vettore che contiene quante volte vengono prese le azioni
N = zeros(A,1);

iterazioni = 100;
count = 0; % contatore delle iterazioni

finalR = 0; % valore del reward alla fine del loop
initialQ = Q;
andamentoQ = zeros(A,iterazioni); % serve per fare i grafici

while (count < iterazioni)

    % Scelta dell'azione con epsilon greedy
    At = epsgreedy(Q,epsilon);
    % azione casuale dell'avversario
    Avs = randi(A); 

    % Calcolo del reward
    Rt = vincitore(At,Avs);
    finalR = finalR + Rt;

    % Incremento le variabili Q e N relative all'azione che ho eseguito
    N(At,1) = N(At,1) + 1;
    Q(At,1) = Q(At,1) + (1/N(At,1))*(Rt-Q(At,1));

    count = count +1;
    for i = 1:A
        andamentoQ(i,count) = Q(i,1);     %Inserimento dei nuovi dati nel vettore
    end


end

figure(1)
stem(N,'filled','-');
grid on
title('Epsilon-greedy algorithm');

time = 0:1:iterazioni;
andamentoQ = horzcat(initialQ,andamentoQ);

figure(2)
plot(time, andamentoQ, LineWidth = 1);
grid on
title('andamento del valore delle azioni Q');
legend('Q(1)','Q(2)','Q(3)','Q(4)','Q(5)');

fprintf("Reward finale per il giocatore 1: ");
disp(finalR);


%% 

% Upper confidence bound optimistic initialization 
clear all
close all
clc

% 1 = Sasso, 2 = Carta, 3 = Forbice, 4 = Spock, 5 = Lizard
rng(812);
A = 5;

% Parametri UCB
alpha = 0.2;
c = 1;

% Inizializzo il vettore delle stime dei valori delle azioni che viene poi
% aggiornato 
Q = 10*ones(A,1);
N = zeros(A,1);

iterazioni = 1000;
count = 0;

finalR = 0;                 %Valore del ritorno alla fine del loop
initialQ = Q;               %Salvo il valore di partenza della variabile Q
andamentoQ = zeros(A,iterazioni);         %Vettore che mantiene traccia di Q nel tempo (usato per il plot)

while (count < iterazioni)
    count = count + 1;                  %Aggiornamento del contatore delle iterazioni
    
    %Scelta dell'azione da prendere all'istante t dai due giocatori
    At = upperconfidencebound(Q,count,N,c);                %Azione presa all'istante t dal giocatore 1     
    ARand = randi(A);                       %Azione presa all'istante t dal giocatore 2

    %Calcolo della ricompensa per il giocatore 1
    Rt = vincitore(At,ARand);         %Ricompensa all'istante t del giocatore 1
    finalR = finalR + Rt;                   %Aggiornamento della ricompensa totale

    %Incremento delle variabili N e Q relative solo all'azione considerata
    %dal giocatore 1, mentre il resto rimane invariato
    N(At,1) = N(At,1) + 1;                      
    Q(At,1) = Q(At,1) + (alpha .* (Rt - Q(At,1)));

    for i = 1:A
        andamentoQ(i,count) = Q(i,1);     %Inserimento dei nuovi dati nel vettore
    end
end

figure(1)
stem(N,'filled','-');
grid on
title('UCB algorithm with optimistic initialization and costant step-size');

time = 0:1:iterazioni;
andamentoQ = horzcat(initialQ,andamentoQ);

figure(2)
plot(time, andamentoQ, LineWidth = 1);
grid on
title('andamento del valore delle azioni Q');
legend('Q(1)','Q(2)','Q(3)','Q(4)','Q(5)');

fprintf("Reward finale per il giocatore 1: ");
disp(finalR);


%%

%Preferenze updates
%------------------
clear
close all
clc


rng(10);

% 1 = Sasso, 2 = Carta, 3 = Forbici, 4 = Spock, 5 = Lizard
A = 5;                      %Numero di azioni possibili

alpha = 0.500;               %Inizializzazione del constant step-size
averageR = 0;                %Inizializzazione della ricompensa media

%Inizializzo il vettore che tiene traccia per ogni azione del numero di
%volte che questa viene presa
N = zeros(A,1);
%Inizializzo il vettore che tiene traccia per ogni azione della sua
%preferenza (inizialmente questo valore è uguale per tutte)
H = zeros(A,1);

%Definisco un numero di azioni sufficientemente grande così da garantire
%che per istanti di tempi grandi le stime dei valori di azione siano uguale
%ai valori delle azioni q(a).
iterazioni = 100000;      %Numero di iterazioni totali
count = 0;                %Contatore delle iterazioni considerate

finalR = 0;                 %Valore del ritorno alla fine del loop
initialH = H;               %Salvo il valore di partenza della variabile Q
andamentoH = zeros(A,iterazioni);         %Vettore che mantiene traccia di Q nel tempo (usato per il plot)

while (count < iterazioni)
    count = count + 1;                  %Aggiornamento del contatore delle iterazioni

    %Scelta dell'azione da prendere all'istante t dai due giocatori
    At = preferenceUpdates(H);              %Azione presa all'istante t dal giocatore 1     
    ARand = randi(A);                       %Azione presa all'istante t dal giocatore 2

    %Calcolo della ricompensa per il giocatore 1
    Rt = vincitore(At,ARand);         %Ricompensa all'istante t del giocatore 1
    finalR = finalR + Rt;                   %Aggiornamento della ricompensa totale

    %Aggiorno la media dei reward ottenuti finora
    averageR = averageR + ((Rt - averageR) ./ count);
    
    %Calcolo della preferenze associate ad ogni azione tramite la
    %distribuzione soft-max (utile per l'aggiornamento del vettore H)
    pi = softmax(H);

    %Aggiornamento del valore di preferenza dell'azione presa all'istante t
    H(At,1) = H(At,1) + (alpha .* (Rt - averageR) .* (1 - pi(At,1)));
    %Aggiornamento del vettore N relativo all'azione presa all'istante t
    N(At,1) = N(At,1) + 1;

    for i = 1:A
        %Aggiornamento del valore di preferenza di tutte le azioni non
        %prese all'istante t
        if (i ~= At)
            H(i,1) = H(i,1) - (alpha .* (Rt - averageR) .* pi(i,1));
        end
    end

    for i = 1:A
        andamentoH(i,count) = H(i,1);     %Inserimento dei nuovi dati nel vettore
    end
end

figure(1)
stem(N,'filled','-');
grid on
title('Preference updates algorithm');

time = 0:1:iterazioni;
andamentoH = horzcat(initialH,andamentoH);

figure(2)
plot(time, andamentoH, LineWidth = 1);
grid on
title('andamento del valore delle preferenze H');
legend('H(1)','H(2)','H(3)','H(4)','H(5)');

fprintf("Reward finale per il giocatore 1: ");
disp(finalR);