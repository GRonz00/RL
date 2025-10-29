function vpi = policy_evaluation(pi, P, R, gamma)

% get the number of states
S = size(P, 1);

% define the transition matrix and the reward matrix for the policy
Ppi = zeros(S,S);
Rpi = zeros(S,1);

% sweep over the states
for s = 1:S
    Ppi(s,:) = P(s,:,pi(s));
    Rpi(s,:) = R(s, pi(s));
end

% evaluate the policy (solve the Bellman equation)
vpi = (eye(S) - gamma*Ppi)\Rpi;