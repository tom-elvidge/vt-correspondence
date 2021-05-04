% Manually define landmarks.
LM_1_n = [
    19.7 284.5 271.11; % back left tooth
    64.169 286.6 273.927; % back right tooth
    44.125 293.932 280.226; % hard palate peak
    44.77 256.378 274.51; % hard palate back
    43.2002 200.857 151.223; % back top of middle glottis
    43.2326 237.572 264.509; % uvula
    41.6472 328.72 261.482; % top middle mouth opening
    42 329 254.342; % bottom middle mouth opening
];
LM_1_c1 = [
    19.7 284.5 271.11; % back left tooth
    64.169 286.6 273.927; % back right tooth
    44.125 293.932 280.226; % hard palate peak
    44.77 256.378 274.51; % hard palate back
    33.0553 200.996 146.755; % back top of middle glottis
    39.5014 230.114 263.415; % uvula
    43.7627 330.477 263.252; % top middle mouth opening
    40.6018 331.326 249.362; % bottom middle mouth opening
];

% Open source and target surface mesh.
TR_1_n = stlread("data/s1_neutralVT_final.stl");
TR_1_c1 = stlread("data/s1_contrast1_final_tr.stl");

% Compute landmark selection matrices.
[TR_1_n_Sx, TR_1_c1_Sy] = SelectionMatrix(TR_1_n.Points, TR_1_c1.Points, LM_1_n, LM_1_c1);

% Landmark deformation.
TRx1 = LBME(TR_1_n, TR_1_n_Sx, TR_1_c1, TR_1_c1_Sy, 2.7);

% Closest point deformation.
TRxn = LBRP(TRx1, TR_1_c1, 0.1, 1);
stlwrite('data/s1_neutralVT_deformed_n.stl', TRxn.ConnectivityList, TRxn.Points);

% Plot deformed source coloured with displacement field.
diff = TRxn.Points - TR_1_n.Points;
norms_1 = sqrt(sum(diff.^2,2));
TRxn_1 = TRxn;
figure(1);
trisurf(TRxn_1, norms_1);
colorbar
caxis([0 25])
% set(gca,'ColorScale','log')

% Plot deformation field
figure(2);
[Sx, Sy] = MutualClosestPoints(TRxn.Points, TR_1_c1.Points);
x = Sx * TR_1_n.Points;
y = (Sy * TR_1_c1.Points) - x;
quiver3(x(:,1),x(:,2),x(:,3),y(:,1),y(:,2),y(:,3),'AutoScale','off');


