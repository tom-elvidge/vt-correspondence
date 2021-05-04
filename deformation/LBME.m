function TRx1 = LBME(TRx0, Sx, TRy, Sy, lambda)

% Extract source and target vertices.
x0 = TRx0.Points;
y = TRy.Points;

% Maintain edges before and after deformation.
fx = TRx0.ConnectivityList;

% Compute deformation from paper.
[L,~] = LaplaceBeltrami(x0', fx'); 
b = [lambda*(L*x0) ; Sy*y];
A = [lambda*L ; Sx];
% Least squares solution for x1.
x1 = A\b;

% Create new mesh using the deformed vertices.
TRx1 = triangulation(fx, x1);

end

