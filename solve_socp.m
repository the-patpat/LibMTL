function [x, exitflag] = solve_socp(v1, v2, n, retain)
    socConstraints = secondordercone(sparse(1:n, 1:n, 1.0), sparse(zeros(n,1)), ...
                                sparse(zeros(n,1)), -norm(v1));
    c_theta = dot(v1,v2)/(norm(v1)*norm(v2));
    theta = acos(c_theta);
    min_gmodx = cos(0.5*pi - retain*theta)*dot(v1, v1);

    [x, fval, exitflag, output, lambda] = coneprog(v1, socConstraints, [-v1';v2'], [-min_gmodx;0], [], []);
end