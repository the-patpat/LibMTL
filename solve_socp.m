function [x, exitflag] = solve_socp(v1, v2, n)
    socConstraints = secondordercone(sparse(1:n, 1:n, 1.0), sparse(zeros(n,1)), ...
                                sparse(zeros(n,1)), -norm(v1));
    [x, fval, exitflag, output, lambda] = coneprog(v1, socConstraints, [-v1';v2'], [0;0], [], []);
end