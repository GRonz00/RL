function At = preferenceUpdates(H)
    %Funzione che definisce l'azione da prendere in accordo con
    %la distribuzione di probabilità soft-max, la quale assegna ad ogni
    %azione un certo range di probabilità di essere scelta. Successivamente
    %si prende casualmente un valore tra 0 ed 1 ed a seconda di questo
    %viene scelta l'azione corrispondente
    Pr = softmax(H);

    %Calcolo la somma cumulativa, ognuna data dal valore corrente di
    %preferenza più quello passato (infatti l'ultima somma vale proprio 1)
    sumPr = cumsum(Pr);

    At = find(rand < sumPr,1,"first");  %Selezione dell'azione migliore