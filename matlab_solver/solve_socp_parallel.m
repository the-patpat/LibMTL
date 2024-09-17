function [x, exitval] = solve_socp_parallel(v1, v2, n_channels, part_size, retain)
    
    % Create parallel pool
    p = gcp;
    % v1 and v2 are already the indexed gradients, i.e. beg/end 
    
    % vec1 = v1(:, :);
    % vec2 = v2(:, :);
    % We have n_channels that we need to compute with part_size parameters each
    for i = n_channels:-1:1
        if dot(v1(1+(i-1)*part_size:i*part_size), ...
                    v2(1+(i-1)*part_size:i*part_size)) > 0
            f(i) = parfeval(p, @solve_socp, 2, v1(1+(i-1)*part_size:i*part_size), ...
                        v2(1+(i-1)*part_size:i*part_size), ...
                        part_size, retain);
        else
            f(i) = parfeval(p, @echo_vector, 2, ...
                v1(1+(i-1)*part_size:i*part_size));
        end
    end
    % while ~all(or(strcmp({f.State}, 'finished'),strcmp({f.State}, 'unavailable')))
        % pause(5)
    % end
    % unavailable = find(strcmp({f.State}, 'unavailable'));
    % for i = unavailable

    % end
    wait(f)
    [x, exitval] = f.fetchOutputs();
    clear f;
    % x = v1;
    % exitval = 1;
end

function [x, exitval] = echo_vector(v)
    x = v;
    exitval=2;
end