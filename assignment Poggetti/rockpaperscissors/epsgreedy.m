function a = epsgreedy(Q, epsilon)
% epsilon greedy action selection

A = size(Q,1);

if rand < epsilon
    % take casual action
    a = randi(A);
else
    % take greedy action
    % parity broken using first index
    % a = find(Q == max(Q), 1, "first");

    % parity broken at random
    optimal_actions = find(Q == max(Q));
    a = optimal_actions(randi(length(optimal_actions)));
end