% 23/6/2021,This is the core solver. 
% This is the script that implements the AM algorithm for our simple 1D
% Tikhonov Signal Denoising. 
%Study this code carefully and relate with all the equations in the slide
%!!!
function out = Tikhonov_1D(b, A, lam, rho, Nit, tol)

%Initializations

x         = b;   
z = b;
N         = length(x);
funcVal   = zeros(Nit,1);
relError = zeros(Nit,1);


[D,DT,DTD] = defDDt(N); % In the slides we use nabla symbol for this (nabla is the upside down triangle symbol)
                        % the T here stands for the transpose.
                        % D = nabla, DT = nabla transpose, DTD = nabla
                        % transpose nabla. Plese relate to the equation of
                        % linear system in slides.

% Slight different notation. Here we use A. but in the slides its the H.
AT = A';     %AT = A trasposed
ATA = A'*A;  % ATA = A transposed A. See where we use this in slide tutorial.

%====================== Main loop of algorithm ======================
for k = 1:Nit
    
    %%%%%% z-subproblem %%%%%%
    %z_old = z;
    z = (rho/(rho + 2*lam))*D*x;
    
    %%%%%% x-subproblem %%%%%%
    % Here is where we solve the linear system.
    
    lhs = ATA + rho*DTD; % left hand side of our linear system. Can you relate this to the equation in slides ?
    rhs = AT*b + rho*DT*z; % Right hand side of the linear system. Relate this to the linear system equation in slides
    
    x_old = x; % store the previous iteration restored signal. To compute the tol
    
    [x,~] = cgs(lhs,rhs,1e-5,10); % AHA ! here we solve the linear system using the conjugate gradient method.
                                   % conjugate gradient method is build in
                                   % MATLAB. Please look into the help
                                   % documents on this cgs function !
    
    
    relError(k)    = norm(x - x_old,'fro')/norm(x, 'fro');
    funcVal(k)   = (1/2)*norm(A*x - b)^2 + (rho/2)*norm(z - D*x)^2 +lam*norm(z)^2;
    
    if relError(k) < tol % If error between current restored signal and previous restored signal is less than tol. Exit program !
        break;
    end
    
    
end


out.sol = x;
out.funVal = funcVal(1:k);
out.relativeError       = relError(1:k);
end



function [D,DT,DTD] = defDDt(N)
%Create a first order difference matrix D (Nabla)
% Please study this code to know what is actually nabla !
e = ones(N,1);
B = spdiags([e -e], [1 0], N, N);
B(N,1) =  1;

D = B; 
clear B;
% Create the transpose of D
DT = D'; %Remember that DT = -D, also called the backward difference.
DTD = D'*D;
end
