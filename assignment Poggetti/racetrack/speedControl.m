function V = speedControl(currentV,newV)
    % funzione che riceve in ingresso le componenti delle velocità ed
    % esegue i controlli del caso 

    % Controllo la negatività di ciascuna componente
    % Componente lungo x negativa
    if (newV(1) < 0 && newV(2) > 0) 
        % Saturo a zero la componente negativa 
        newV(1) = 0;
    % Componente lungo y negativa
    elseif (newV(1) > 0 && newV(2) < 0)
        newV(2) = 0;
    % Nel caso siano entrambe negative o nulle la nuova velocità rimane
    % uguale alla precedente
    elseif (newV(1) <= 0 && newV(2) <= 0)
        newV = currentV;
    end
    
    % Controllo sul valore massimo 
    if(newV(1) > 4)
        % Saturo la componente lungo x a 4 (max)
        newV(1) = 4;
    elseif(newV(2) > 4)
        % Saturo la componente lungo y a 4 (max)
        newV(2) = 4;
    end

    % Velocità finale
    V = newV;
end