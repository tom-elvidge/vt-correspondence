function [Sx, Sy] = SelectionMatrix(x, y, LMx, LMy)

% Find landmarks on the surfaces.
LMx_idx = knnsearch(x, LMx, 'K', 1);
LMy_idx = knnsearch(y, LMy, 'K', 1);

% Manually select corresponding landmark vertices in x and y.
Sx = zeros(2, size(x, 1));
Sy = zeros(2, size(y, 1));
i = 1;
while i < length(LMx_idx)
    Sx(i, LMx_idx(i)) = 1;
    Sy(i, LMy_idx(i)) = 1;
    i = i + 1;
end
end

