function pip = policy_improvement(vpi, P, R, gamma)

% get the number of states and actions
S = size(P, 1);
A = size(R, 2);

% construct the quality function 
Q = zeros(S,A);
% construct the improved policy
pip = zeros(S,1);

% sweep over the states and actions
for s = 1:S
    for a = 1:A
        % compute the quality function
        Q(s,a) = R(s,a) + gamma*P(s,:,a)*vpi;
    end

    if(s == 1 || s == S)

        pip(s) = 1;
    else
        % find the policy greedy wrt the quality function
        pip(s) = find(Q(s,:) == max(Q(s,:)), 1, "first");
    end

end