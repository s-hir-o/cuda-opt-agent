"""Build instruction dependency graph from parsed PTX instructions.

Computes RAW/WAR/WAW dependencies, critical path length, and ILP estimate.
"""

from __future__ import annotations

from cuda_opt_agent.schemas.analysis import (
    DependencyEdge,
    DependencyGraph,
    DepType,
    PTXInstruction,
)


def build_dependency_graph(instructions: list[PTXInstruction]) -> DependencyGraph:
    """Build the full dependency graph for a sequence of PTX instructions.

    Args:
        instructions: Ordered list of parsed PTX instructions.

    Returns:
        DependencyGraph with edges, critical path length, and ILP estimate.
    """
    edges: list[DependencyEdge] = []
    n = len(instructions)

    # Track most recent write and read index per register
    last_write: dict[str, int] = {}
    last_read: dict[str, int] = {}

    for instr in instructions:
        idx = instr.index

        # RAW: this instruction reads what an earlier instruction wrote
        for reg in instr.src_registers:
            if reg in last_write:
                edges.append(DependencyEdge(
                    from_idx=last_write[reg],
                    to_idx=idx,
                    dep_type=DepType.RAW,
                    register=reg,
                    latency=1,
                ))

        # WAR: this instruction writes what an earlier instruction reads
        for reg in instr.dest_registers:
            if reg in last_read:
                edges.append(DependencyEdge(
                    from_idx=last_read[reg],
                    to_idx=idx,
                    dep_type=DepType.WAR,
                    register=reg,
                    latency=0,
                ))

        # WAW: this instruction writes what an earlier instruction wrote
        for reg in instr.dest_registers:
            if reg in last_write:
                edges.append(DependencyEdge(
                    from_idx=last_write[reg],
                    to_idx=idx,
                    dep_type=DepType.WAW,
                    register=reg,
                    latency=0,
                ))

        # Update tracking
        for reg in instr.src_registers:
            last_read[reg] = idx
        for reg in instr.dest_registers:
            last_write[reg] = idx
            last_read.pop(reg, None)  # write kills pending reads

    critical_path = _compute_critical_path(n, edges)
    ilp = n / critical_path if critical_path > 0 else float(n)

    return DependencyGraph(
        edges=edges,
        num_instructions=n,
        critical_path_length=critical_path,
        estimated_ilp=ilp,
    )


def _compute_critical_path(num_instructions: int, edges: list[DependencyEdge]) -> int:
    """Compute longest dependency chain using dynamic programming.

    Only RAW edges have non-zero latency and contribute to the critical path.
    Sequential instruction ordering is implicit (index i → i+1 with latency 0).
    """
    if num_instructions == 0:
        return 0

    dist = [0] * num_instructions

    # Process edges sorted by from_idx (topological order since indices are sequential)
    for e in sorted(edges, key=lambda e: (e.from_idx, e.to_idx)):
        new_dist = dist[e.from_idx] + e.latency
        if new_dist > dist[e.to_idx]:
            dist[e.to_idx] = new_dist

    # Also enforce sequential ordering: instruction i+1 can't start before i
    for i in range(1, num_instructions):
        if dist[i] < dist[i - 1]:
            dist[i] = dist[i - 1]

    return max(dist) if dist else 0
