clear all
close all
clc

% import the data of the problem
load gambler_MDP.mat

% tolerance
toll = 1e-4;

% discount factor
gamma = 0.9;

% get the number of states and actions
S = size(P,1);
A = size(R,2);

% initialize the value function
vstar = randn(S,1);
vstar(1,1) = 0;
vstar(S,1) = 0;

% counter for the number of iterations
count = 0;
% iterate until convergence
while true
    % increment the counter
    count = count + 1;
    % update the value function
    vstarp = value_iteration_step(P, R, gamma, vstar);
    % if not updated, break
    if norm(vstarp-vstar,inf) < toll
        break;
    else
        vstar = vstarp;
    end
end

% find the optimal policy as the one that is greedy wrt vstar
pistar = policy_improvement(vstar, P, R, gamma);

% % plot the obtained policy and value function
visualize_policy_value(vstar, pistar)

% save the value function
save vs_vi.mat vstar pistar