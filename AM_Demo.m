
% Created on 18/6/2021 by Dr.Tarmizi Adam.
% Small demo for 1D-signal Total Variation (TV) denoising.
% The algorithm used for minimizing the TV functional is the Alternating 
% Minimization (AM) algorithm. Also known as block coordinate descent or
% Gauss-Siedel method.
% This demo file calls the function "AM_1D()" solver.

clc;
clear all;
close all;





load testSig3.mat; % Load our signal
                    % You can use any  signal that you have

x = testSig3;
x = x';
N = length(x);

%add some noise to it
sigma = 10; % Noise level, the larger, the more contaminated.
A = eye(N); % For signal or image denoising. the matrix A will be an identity matrix
b = A*x + sigma * randn(length(x),1); % Image/signa degradation model. b = Ax + n


%% ********** Global parameter initialization to all the algorithms*******
Nit = 500; % AM is iterative algorithm. We set number of AM iteration to be 500.

%% Parameters for AM TV Denoising
mu = 0.075; % the parameter mu. playing around with this will have effect on denoising
rho   = 1.5; % Value of rho. For now, just let this be 1.5
tol   = 1e-6; % This is used to stop the AM algorithm. If error between 
              % the denoised signal at the current iteration and the
              % previous iteration is tol amount, we exit the AM algorithm.

%% ********** Run the TV-solver ***************

out = AM_1D(b, A, mu, rho, Nit, tol); %Run the AM TV denoising !!!

%% ********************************************

rmseAM = sqrt(mean((x-out.sol).^2)); % Root mean square error. Google This !
%%

%% Some plotting options to show our results

fopt  = min(out.funVal);

figure;
subplot(3,1,1)
plot(x,'LineWidth',2.5);
title('\textbf{Original signal} $\textbf{x}$','interpreter','latex','FontSize', 15);
set(gca, 'FontSize', 15);
axis tight

subplot(3,1,2);
plot(b,'LineWidth',2.5)
title('\textbf{Noisy signal} $\textbf{b}$','interpreter','latex','FontSize', 15);
set(gca, 'FontSize', 15);
axis tight

subplot(3,1,3);
plot(out.sol,'LineWidth',2.5);
title('\textbf{AM TV denoised}','interpreter','latex','FontSize', 15);
set(gca, 'FontSize', 15);
axis tight


figure;
subplot(2,1,1)
semilogy(out.funVal - fopt,'Linewidth',3.5,'Color','blue'); hold;
xlabel('Iterations','FontSize',25,'Interpreter', 'latex');
ylabel('$F(x) - F*$','FontSize',25,'Interpreter', 'latex');
l = legend('Alternating Minimization TV');
set(l,'interpreter','latex','FontSize', 15);
axis tight;
grid on;
set(gca, 'FontSize', 15);

subplot(2,1,2)
semilogy(out.relativeError,'Linewidth',3.5,'Color','blue');hold;
xlabel('Iterations','FontSize',25,'Interpreter', 'latex');
ylabel('Relative Err','FontSize',25,'Interpreter', 'latex');
l = legend('Alternating Minimization TV');
set(l,'interpreter','latex','FontSize', 15);
axis tight;
grid on;
set(gca, 'FontSize', 15);




%%