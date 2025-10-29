function [nextS, R, v ] = updateState(state, v, track, initialState, finalState)

    % Viene restituito in uscita il nuovo stato in cui si posiziona l'auto,
    % la ricompensa ottenuta e la velocità dell'auto
    initialS = length(initialState);
    finalS = length(finalState);

    %Ricavare gli indici di riga e di colonna dello stato corrente
    [numRow, numCol] = ind2sub([32 17], state);

    % Calcolo stato successivo e faccio il controllo se sia o meno uno
    % stato possibile
    % Attenzione : in matlab le righe crescono verso il basso
    nextRow = numRow - v(1); 
    nextCol = numCol + v(2);

    if (nextRow < 1 || nextCol < 1 || nextRow >32 || nextCol > 17)
        %Condizione in cui la transizione porta l'auto fuori dalla matrice,
        %quindi non sono operazioni possibili

        %Riposizionamento casuale dell'auto in uno degli stati di partenza, 
        %assegnazione negativa della ricompensa e reset della velocità
        nextS = initialState(randi(initialS));
        % nextState = initialState(1);
        R = -1;
        v = [0,0];
    else
        % Auto si sposta all'interno della matrice degli stati
        nextS = sub2ind([32 17],nextRow,nextCol);
        
        % Controllo se il nuovo stato faccia parte della pista o meno
        if(track(nextS) == 0) % Fuori pista
            nextS = initialState(randi(initialS));
            R = -1;
            v = [0 0];
        else
            if (nextS >= finalState(1) && nextS <= finalState(finalS))
                % Raggiunto la fine 
                R = 0;
                v = [0 0];
            else
                R = -1;
            end
        end
    end

end