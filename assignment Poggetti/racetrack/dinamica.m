clear all
close all
clc 

% Numero di stati 
S = 17*32;

% Matrice della pista
track = zeros(32,17);

for s = 5:14
    track(s) = 1;
end

for s = 36:54
    track(s) = 1;
end

for s = 66:93
    track(s) = 1;
end

for s = 97:295
    track(s) = 1;
end

%Usando i seguenti indici si considerano le coordinate dello stato i-esimo
for j = 11:17
    for i = 1:6
        track(i,j) = 1;
    end
end

% Visualizzazione della pista
imagesc(track);
colormap("parula");
axis equal;
title('Race Track');
xlabel('Track Width');
ylabel('Track Length');

%Inizializzazione degli stati di partenza e degli stati di arrivo
initialState = [128; 160; 192; 224; 256; 288];
finalState = [513; 514; 515; 516; 517; 518];

save raceTrack.mat  S track initialState finalState

