function index = multinomial_resample(weight,u)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Multinomial resampling using pre-drawn uniforms.

[u,sort_order] = sort(u(:));
cumulative = cumsum(weight);
cumulative(end) = 1;
M = length(u);
sorted_index = zeros(M,1);

j = 1;
for i = 1:M
    while u(i) > cumulative(j)
        j = j+1;
    end
    sorted_index(i) = j;
end

index = zeros(M,1);
index(sort_order) = sorted_index;
end
