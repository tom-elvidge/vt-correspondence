function [Sx, Sy] = MutualClosestPoints(x, y)
% Find mutual closest points between x and y. Returns selection matrices
% where Sx is (length(x), n) and Sy is (length(y), n) where n is the number
% of mutual correspondences. If Sx(xi, j) = 1 and Sy(yi, j) = 1 then the
% points x(xi,:) and y(yi,:) are mutual cloeset point correspondences.

% Get nearest neighbours in x of each y and vice versa.
xnn = knnsearch(x, y, 'K', 1);
ynn = knnsearch(y, x, 'K', 1);

% Correspondence matrix. C(i)=j means y(i) corresponds to x(j).
C = length(y);
yi = 1;
while yi < length(y) + 1
    % Get the nearest neighbour index in x of yi.
    xi = xnn(yi);
    % Get the nearest neighbour index in y of xi.
    yi2 = ynn(xi);
    % If yi and yi2 are the same index then it is a mutual closest point.
    if yi == yi2
        C(yi) = xi;
    end
    yi = yi + 1;
end

LMy = find(C~=0);
LMx = C(LMy);
n = length(LMy);

% Create selection matrices.
Sx = zeros(n, length(x));
Sy = zeros(n, length(y));
i = 1;
while i < n + 1
    Sx(i, LMx(i)) = 1;
    Sy(i, LMy(i)) = 1;
    i = i + 1;
end

end

