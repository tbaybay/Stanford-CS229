function sq_distance = l2_squared(A, B)
    % Returns a matrix of distances between the mth vector of A and nth
    % vector of B
    sq_distance = zeros(size(A, 2), size(B, 2));
    for m = 1:size(A, 2)
        for n = 1:size(B, 2)
            sq_distance(m, n) = norm(A(:, m) - B(:, n)).^2;
        end
    end
end