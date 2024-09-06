def shortest_path_relaxation(gcs, xv, zv, ze_out, ze_inc, s, t):

    yv = gcs.vertex_relaxed_binaries()
    ye = gcs.edge_relaxed_binaries()

    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_indices(vertex)
        out_edges = gcs.outgoing_indices(vertex)

        constraints.append(yv[i] <= 1)
        # 确保索引 i 在 ye 的范围内
        if i < len(ye):
            constraints.append(ye[i] <= 1)
        else:
            print(f"Warning: Index {i} is out of bounds for ye with size {len(ye)}")
        
        if vertex == s:
            constraints.append(sum(ye[inc_edges]) == 0)
            constraints.append(sum(ye[out_edges]) == 1)
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == xv[i])
            
        elif vertex == t:
            constraints.append(sum(ye[out_edges]) == 0)
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints.append(zv[i] == xv[i])
            
        else:
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(yv[i] <= 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints += vertex.conic.eval_constraints(xv[i] - zv[i], 1 - yv[i])
    
    return constraints
