function ix = knn(distances, j, K)
    % Returns the indices of the K smallest values in the jth row of distances 
    ix = zeros(K, 1);
    d = distances(j, :);
    d(j) = [];
    for k = 1:K
        [~, ix(k)] = min(d);
        d(ix(k)) = [];
    end
end