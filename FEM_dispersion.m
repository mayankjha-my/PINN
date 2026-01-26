%% SAFE-PML Dispersion (k–omega diagram)
clear; clc;
syms x y

% --- Parameters (set your values here) ---
mu44l = 4.35*10^9; mu66l = 5*10^9; mu44h = 5.3*10^9; mu66h = 6.47*10^9;
rho1 = 9890; rho2 = 3400; g = 9.81;
H =100; beta1 = 0.05; beta2 = 0.05; P1 = 10^9; P2 = 10^9;
v = 7+7i; phi = pi/6; mu_eH02=0.3*mu44l;
l=29*H;
J1 = H/2;  % Jacobian


beta_1 = sqrt(mu44l/rho1);
w=(x*y*beta_1/H);
k=(x/H);

% --- Shape functions (quadratic, 3-node) ---
chi_fun = @(xi) [ (xi.*(xi-1))/2 ; (1 - xi.^2) ; (xi.*(xi+1))/2 ];
dchi_fun = @(xi) [ (2*xi-1)/2 ; -2*xi ; (2*xi+1)/2 ];

% --- Gauss-Legendre quadrature (3-point) ---
gp = [-sqrt(3/5), 0, sqrt(3/5)];
gw = [5/9, 8/9, 5/9];

% === Element 1 matrices ===
Ke1 = sym(zeros(3,3)); 
Me1 = sym(zeros(3,3));
for ig = 1:3
    xig = gp(ig); wg = gw(ig);
    Ni = chi_fun(xig);
    dNi = dchi_fun(xig);
    wfac1 = (1 + sin(beta1*H*(xig+1)/2));

    for i = 1:3
        for j = 1:3
            term1 = -k^2*mu66l*Ni(i)*Ni(j);
            term2 = (1/J1^2)*mu44l*dNi(i)*dNi(j);
            term3 = (P1/2)*k^2*Ni(i)*Ni(j);
            term4 = mu_eH02*(-k^2*(cos(phi))^2*Ni(i)*Ni(j) ...
                     + 1i*k*sin(2*phi)/J1*Ni(i)*dNi(j) ...
                     + (sin(phi))^2/J1^2*dNi(i)*dNi(j));
            Ke1(i,j) = Ke1(i,j) + (wfac1*(term1+term2+term3) + term4)*J1*wg;
            Me1(i,j) = Me1(i,j) + rho1*wfac1*Ni(i)*Ni(j)*J1*wg;
        end
    end
end

% === Element 2 matrices ===
Ke2 = sym(zeros(3,3)); 
Me2 = sym(zeros(3,3));
J1=l/2;
for ig = 1:3
    xig = gp(ig); wg = gw(ig);
    Ni = chi_fun(xig);
    dNi = dchi_fun(xig);
    wfac2 = (1 - sin(beta2*H*(xig+1)/2));

    for i = 1:3
        for j = 1:3
            term1 = -k^2*mu66h*Ni(i)*Ni(j);
            term2 = (1/(J1^2*v^2))*mu44h*dNi(i)*dNi(j);
            term3 = (P2/2)*k^2*Ni(i)*Ni(j);
            term4 = (k^2*rho2*wfac2*g/2)*(H*(xig+1)/2)*Ni(i)*Ni(j);
            term5 = -(rho2*wfac2*g/(2*J1*v))*(dNi(j)*Ni(i));
            term6=  -(rho2*g/(2*J1*v))* ( -beta2*sin(beta2*H*(xig+1)/2)+ wfac2)*(dNi(i)*Ni(j));
            term7 = -(rho2*wfac2*g/(2*J1^2*v^2))*(H*(xig+1)/2)*dNi(i)*dNi(j);
            Ke2(i,j) = Ke2(i,j) + (wfac2*(term1+term2+term3) + term4+term5+term6+term7)*v*J1*wg;
            Me2(i,j) = Me2(i,j) + rho2*wfac2*Ni(i)*Ni(j)*v*J1*wg;
        end
    end
end

% === Global assembly ===
K = sym(zeros(5,5)); 
M = sym(zeros(5,5));
nodes1 = [1 2 3];
nodes2 = [3 4 5];
K(nodes1,nodes1) = K(nodes1,nodes1) + Ke1;
M(nodes1,nodes1) = M(nodes1,nodes1) + Me1;
K(nodes2,nodes2) = K(nodes2,nodes2) + Ke2;
M(nodes2,nodes2) = M(nodes2,nodes2) + Me2;

% Eliminate DOF 5 (v5=0)
K(5,:) = [];  K(:,5) = [];
M(5,:) = [];  M(:,5) = [];

% === Dispersion determinant ===
A = K - w^2*M;
det_A = real(det(A));

% === Plot dispersion relation ===
fimplicit(det_A, [1,3,-3,3], 'k', 'LineWidth', 2)
xlabel('kH')
ylabel('$c/c_s$','Interpreter','latex')

title('SAFE-PML Dispersion Relation')
grid on
% [0.5,2,0,3]