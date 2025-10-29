function vstarp = value_iteration_step(P, R, gamma, vstar)

% get the number of states and actions
S = size(P, 1);
A = size(R, 2);

% construct the quality function 
Q = zeros(S,A);
% construct the updated value function
vstarp = zeros(S,1); 

% sweep over the states and actions
for s = 2:S-1
    for a = 1:A
        % compute the quality function
        Q(s,a) = R(s,a) + gamma*(squeeze(P(s,:,a)))*vstar;
    end
    % use the optimality Bellman equation as update
    vstarp(s) = max(Q(s,:));
    vstarp(1) = 0;
    vstarp(S) = 0;
end