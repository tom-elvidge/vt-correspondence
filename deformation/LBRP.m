function TRxn = LBRP(TRx, TRy, lambda, n)

TRxi = TRx;

i = 1;
while i < n + 1
    % Get mutually closest points.
    if length(TRxi.Points) >= length(TRy.Points)
        [Sx, Sy] = MutualClosestPoints(TRxi.Points, TRy.Points);
    else
        [Sy, Sx] = MutualClosestPoints(TRy.Points, TRxi.Points);
    end
    
    % Find deformation.
    TRxi = LBME(TRxi, Sx, TRy, Sy, lambda);
    
    % Next iteration.
    i = i + 1;
end

TRxn = TRxi;

end

