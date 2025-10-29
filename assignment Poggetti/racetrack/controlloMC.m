clear all
close all
clc

% Carico le informazioni riguardanti la mappa e gli stati iniziali e finali
load raceTrack.mat

% numero di episodi totali
numEp = 5e5;
% Stati terminali
finalS = length(finalState);
% Stati iniziali
initialS = length(initialState);

epsilon = 0.8;
decrease = 0.0002;
gamma = 0.9;
alpha = 0.01;

% Inizializzazione della matrice delle azioni possibili
[vx, vy] = meshgrid(-1:1, -1:1);
actionList = [vx(:), vy(:)];

A = length(actionList); 


%FASE DI INIZIALIZZAZIONE DELL'ALGORITMO
Q = zeros(S,A);                        %Stima della funzione qualità
N = zeros(S,A);                        %Occorrenze per ogni coppia stato-azione
pi = randi(A,[S 1]);                   %Policy iniziale randomica

pi(finalState) = 5; % indice che corrisponde all'azione [0 0] poiché sono arrivato in uno stato terminale

counters = [];                         %Contiene tutti i contatori associati ad ogni singolo episodio
bestPath = [];                         %Contiene tutti gli stati percorsi dall'auto usando la policy ottimale
bestActions = [];                      %Vettore delle azioni prese nel percorso migliore
valBestPath = inf;%Indica il numero di stati percorsi dall'auto usando la policy ottimale


for i = 1:numEp

    fprintf("Episodio numero: ");
    disp(i);
    fprintf("Epsilon utilizzato:");
    disp(epsilon);
    
    %Scelta casuale dello stato iniziale
    S0 = initialState(randi(initialS));

    % Variabile booleana per la prima azione di ciascun episodio
    first = true;
    % Per ogni episodio inizialmente componenti delle velocità nulle
    v = [0, 0];

    % Inizializzazione vettori che tengono traccia degli stati azioni e
    % rewards per l'episodio corrente
    states = S0;
    currentState = S0;
    actions = [];
    rewards = [];

    % Valore del reward per ogni passo nella pista
    R = -1;
    % Contiene il numero di iterazioni per arrivare nello stato terminale 
    counter = 1;

    while ( R == -1)
        % Se è la prima azione dell'episodio corrente
        if (first)
            % Scegli azione di partenza casuale
            a = randi(A);
            % Azione vera e propria
            action = actionList(a,:);
            
            % Ora metto a zero la variabile booleana perché ho scelto la
            % prima azione
            first = false;
        else
            if (rand(1) < epsilon)
                a = randi(A);
                action = actionList(a,:);
            else
                % Azione scelta seguendo la policy
                a = pi(currentState);
                action = actionList(a,:);
            end
        end

        % Ora devo calcolare il nuovo stato in base alla velocità
        newV = v + action;
        % Effettuo un controllo sulle velocità
        v = speedControl(v,newV);

        % Calcolo il prossimo stato 
        [nextS , R , v] = updateState(currentState,v,track,initialState,finalState);

        % Aggiungo i nuovi valori alle varie liste
        states  = [states, nextS];
        rewards = [rewards, R];
        actions = [actions, a];

        % Aggiorno variabili per iterazione successiva
        currentState = nextS;
        counter = counter + 1;


    end

    % EPISODIO i-ESIMO TERMINATO

   

    counters = [counters,counter];

    % Aggiornamento parametro epsilon
    if(mod(i,100) == 0 && epsilon > 0.05)
        epsilon = epsilon - decrease;
    end

    %Se l'episodio generato in questa iterazione è migliore di tutti gli
    %altri generati fino ad ora allora lo salvo come "miglior percorso"
    if (states(end) >= finalState(1) && states(end) <= finalState(finalS) && counter < valBestPath)
        bestPath = states;                       %Aggiornamento del percorso migliore
        valBestPath = counter;                   %Aggiornamento del numero di stati del percorso migliore
        bestActions = actions;                   %Aggiornamento delle azioni prese nel percorso migliore
    end

     %Se l'episodio generato in questa iterazione è migliore di tutti gli
    %altri generati fino ad ora allora lo salvo come "miglior percorso"
    if (states(end) >= finalState(1) && states(end) <= finalState(finalS) && counter < valBestPath)
        bestPath = states;                       %Aggiornamento del percorso migliore
        valBestPath = counter;                   %Aggiornamento del numero di stati del percorso migliore
        bestActions = actions;                   %Aggiornamento delle azioni prese nel percorso migliore
    end


    % AGGIORNAMENTO POLICY
    % Si calcola il ritorno atteso G a ritroso, per poter calcolare la
    % funzione qualià e migliorare poi la policy
    G = 0;
    for t = length(rewards):-1:1
        G = rewards(t) + (gamma * G);
        Q(states(t),actions(t)) = Q(states(t),actions(t)) + (alpha * (G- Q(states(t),actions(t))));

        % policy aggiornata
        pi(states(t)) = find(Q(states(t),:) == max(Q(states(t),:)), 1,"first");
    end

end


%% PARTE DI GRAFICI
%% 1) Learning curve: passi per episodio + media mobile
figure; 
plot(counters, '.'); hold on;
win = 1000;                            % finestra media mobile
ma  = movmean(counters, [win-1 0]);    % media mobile causale
plot(ma, 'LineWidth', 1.5);
grid on; xlabel('Episodio'); ylabel('Passi'); 
title('Learning curve (passi per episodio)');
legend('Episodio','Media mobile');

%% 2) Trend di epsilon (se lo aggiorni ogni 100 episodi come nel tuo codice)
% Se non hai salvato la serie, ricostruiscila:
eps0 = 0.8; dec = 0.002; minEps = 0.05;
eps_series = zeros(1,numEp);
e = eps0;
for k = 1:numEp
    if mod(k,100)==0 && e>minEps
        e = max(minEps, e - dec);
    end
    eps_series(k) = e;
end
figure; plot(eps_series, 'LineWidth', 1.2); grid on;
xlabel('Episodio'); ylabel('\epsilon'); title('Andamento di \epsilon');

%% 3) Istogramma lunghezze episodio
figure; histogram(counters, 'BinMethod','sturges');
grid on; xlabel('Passi'); ylabel('Frequenza');
title('Distribuzione lunghezze episodio');

%% Utility per coordinate (r,c) <-> stato lineare
[nRows,nCols] = size(track);
rc2s = @(r,c) sub2ind([nRows nCols], r, c);
s2rc = @(s) ind2sub([nRows nCols], s);

%% 4) Heatmap di V(s) = max_a Q(s,a) (mascherata sulla pista)
V = max(Q, [], 2);                     % Sx1
Vimg = nan(nRows,nCols);               % metti NaN fuori pista
for s = 1:numel(V)
    [r,c] = ind2sub([nRows nCols], s);
    if track(r,c)
        Vimg(r,c) = V(s);
    end
end
figure;
imagesc(Vimg); axis equal tight; 
colormap('parula'); colorbar;
title('Value function V(s) sulla pista');
xlabel('Colonne (x)'); ylabel('Righe (y)');

%% 5) Campo di frecce della policy greedy (solo sulle celle di pista)
% Ricava una policy greedy da Q (se vuoi ignorare quella esplorativa)
[~, pi_greedy] = max(Q, [], 2);

% Mappa indice azione -> vettore (dx,dy) coerente col tuo actionList
% (meshgrid(-1:1,-1:1) produce in quest'ordine:
% [-1,-1; 0,-1; 1,-1; -1,0; 0,0; 1,0; -1,1; 0,1; 1,1])
actionList = [-1 -1; 0 -1; 1 -1; -1  0; 0  0; 1  0; -1  1; 0  1; 1  1];

[X,Y] = meshgrid(1:nCols, 1:nRows);    % coordinate di griglia (colonne=x, righe=y)
U = zeros(nRows,nCols); Vv = zeros(nRows,nCols);  % componenti frecce

for s = 1:nRows*nCols
    [r,c] = ind2sub([nRows nCols], s);
    if track(r,c)
        aIdx = pi_greedy(s);
        dv = actionList(aIdx,:);       % [dx, dy] in convenzione colonne,righe
        U(r,c)  = dv(1);               % spostamento in x (colonne)
        Vv(r,c) = -dv(2);              % segno meno per orientare verso l'alto quando dy>0
    else
        U(r,c) = NaN; Vv(r,c)=NaN;     % niente frecce fuori pista
    end
end

figure; 
imagesc(track); colormap(gray);
hold on; axis equal tight; 
title('Policy greedy: campo di azioni'); xlabel('Colonne (x)'); ylabel('Righe (y)');


% scala frecce per leggibilità
scale = 0.5; 
quiver(X, Y, U, Vv, scale, 'LineWidth',1, 'MaxHeadSize', 2, 'AutoScale','off'); 
hold off;

%% 6) Traiettoria migliore sovrapposta alla pista
figure;
imagesc(track); colormap(gray); axis equal tight; 
title(sprintf('Miglior traiettoria (passi = %d)', valBestPath));
xlabel('Colonne (x)'); ylabel('Righe (y)'); hold on;

% converti bestPath (indici lineari) in (r,c) e plottalo
[rp, cp] = arrayfun(@(s) ind2sub([nRows nCols], s), bestPath);
plot(cp, rp, '-o', 'LineWidth',1.5, 'MarkerSize',3);
legend('traiettoria');

%% (Opzionale) Distribuzione delle azioni usate nel best path
if ~isempty(bestActions)
    figure; histogram(bestActions, 1:A+1);
    grid on; xlabel('Indice azione'); ylabel('Frequenza');
    title('Azioni nel best path');
end


%% --- QUIVER POLICY + TRAIETTORIA MIGLIORE ---

[nRows, nCols] = size(track);

% 1) Policy greedy dagli attuali Q
[~, pi_greedy] = max(Q, [], 2);

% 2) Prepara campo vettori U,V (componenti frecce in x,y)
[X, Y] = meshgrid(1:nCols, 1:nRows);
U = nan(nRows, nCols);
V = nan(nRows, nCols);

for s = 1:nRows*nCols
    [r, c] = ind2sub([nRows nCols], s);
    if track(r,c)
        aIdx = pi_greedy(s);
        dv = actionList(aIdx, :);  % [dx, dy] (colonne, righe)
        if any(dv)                 % opzionale: salta l'azione [0 0] per non sporcare il plot
            U(r,c) = dv(1);
            V(r,c) = -dv(2);       % IMPORTANTISSIMO: YDir normal ⇒ inverti il segno per disegno
        end
    end
end

% 3) Figura: pista + quiver
figure; 
imagesc(track); colormap(gray); axis equal tight;
title('Policy greedy (quiver) e traiettoria'); 
xlabel('Colonne (x)'); ylabel('Righe (y)');
hold on;
scale = 0.5; 
quiver(X, Y, U, V, scale, 'LineWidth', 1.0, 'MaxHeadSize', 2, 'AutoScale', 'off', 'Color', 'r');

% 4) Traiettoria migliore: usa bestPath se disponibile, altrimenti rollout greedy
plot_this_path = [];
if exist('bestPath','var') && ~isempty(bestPath)
    plot_this_path = bestPath(:).';
else
    % --- rollout greedy da uno start valido ---
    maxSteps = 2000;
    v = [0 0];
    % scegli uno start a caso tra quelli consentiti
    s0 = initialState(randi(numel(initialState)));
    path = s0;
    s = s0;
    Rstep = -1;
    steps = 0;
    while (Rstep == -1) && steps < maxSteps
        steps = steps + 1;
        aIdx = pi_greedy(s);
        dv = actionList(aIdx,:);          % [dx, dy]
        newV = v + dv;
        v = speedControl(v, newV);        % <-- tua funzione
        [sNext, Rstep, v] = updateState(s, v, track, initialState, finalState);  % <-- tua funzione
        path(end+1) = sNext;
        s = sNext;
    end
    plot_this_path = path;
end

% 5) Disegna la traiettoria
if ~isempty(plot_this_path)
    [rp, cp] = arrayfun(@(ss) ind2sub([nRows nCols], ss), plot_this_path);
    plot(cp, rp, '-o', 'LineWidth', 1.8, 'MarkerSize', 3, 'Color', [0 0.4470 0.7410]); % blu default
    legend('policy (quiver)','traiettoria', 'Location','bestoutside');
else
    legend('policy (quiver)', 'Location','bestoutside');
end

hold off;
