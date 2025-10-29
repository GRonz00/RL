function vpi = iterative_policy_evaluation(pi, P, R, gamma, vpi0)

% get the number of states
S = size(P, 1);
% define the tolerance
toll = 1e-6;

% define the transition matrix and the reward matrix for the policy
Ppi = zeros(S,S);
Rpi = zeros(S,1);

% sweep over the states
for s = 1:S
    Ppi(s,:) = P(s,:,pi(s));
    Rpi(s,:) = R(s, pi(s));
end

% evaluate the policy (use the Bellman equation as a fixed point iteration)
vpi = Rpi + gamma*Ppi*vpi0;
while norm(vpi - vpi0, Inf) > toll
    vpi0 = vpi;
    vpi = Rpi + gamma*Ppi*vpi0;
end

vpi(1,1) = 0;
vpi(S,1) = 0;