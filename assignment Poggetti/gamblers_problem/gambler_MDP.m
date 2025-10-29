close all
clear all
clc

maxWin = 100; % denaro massimo depositabile

S = maxWin + 1;
A = maxWin - 1;

% probabilità che la monetina faccia testa o croce
prob = 0.5;

% matrice delle probabilità di transizione
P = zeros(S,S,A);

for s = 1:S

    [row,col] = ind2sub([S S],s);
    % denaro effettivo
    row = row - 1;

    for a = 1:A

        if (row == 0 || row == maxWin)
            % mi trovo in uno stato terminale
            % torno sempre nello stesso stato indipendentemente dall'azione
            row = row + 1;
            % calcolo stato successivo
            sp = sub2ind([S S],row,col); 

            % Ora dato che l'unica transizione possibile è quella di
            % tornare nello stesso stato, aggiorno la matrice di
            % transizione

            P(s,sp,a) = 1;
        else
            % Mi trovo in uno stato non terminale, perciò con
            % equiprobabilità del 50% posso andare in 2 stati diversi

            % la scommessa non è altro che il minimo tra quello che ho nel
            % deposito e l'azione a

            scommessa = min(row,a); % serve per evitare di scommettere piu soldi di quanti se ne hanno

            % in caso di vittoria calcolo il nuovo stato
            newRow1 = min((row+1)+scommessa,S);

            % in caso di sconfitta
            newRow2 = max((row+1)-scommessa,1);

            % calcolo del nuovo stato
            sp1 = sub2ind([S S],newRow1,col);
            sp2 = sub2ind([S S],newRow2,col);

            % aggiornamento della matrice di probabilità
            P(s,sp1,a) = prob;
            P(s,sp2,a) = prob;

         
        end
    end

end

%% Matrice dei reward

immediate_r = zeros(S,1);

for s = 1:S
    if (s == 1) % corrisponde ad avere 0 nel deposito
        immediate_r(s,1) = -1;
    elseif (s == S)
        immediate_r(s,1) = 1;
    end
end

R = zeros(S,A);
for a=1:A
    R(:,a) = P(:,:,a)*immediate_r;
end

R(1,:) = 0;
R(S,:) = 0;

save gambler_MDP.mat P R 