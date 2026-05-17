from __future__ import annotations
from math import inf
from planning.pddl import ActionSchema, State, Objects, get_all_groundings, get_applicable_actions


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###

    unsatisfied = frozenset(goal - state)

    if not unsatisfied:
        return 0

    actions = get_all_groundings(domain, objects)

    useful_covers = []
    for action in actions:
        cover = frozenset(action.add_list & unsatisfied)
        if cover:
            useful_covers.append(cover)

    if not useful_covers:
        return inf

    queue = [(unsatisfied, 0)]
    visited = {unsatisfied}

    while queue:
        remaining, cost = queue.pop(0)

        if not remaining:
            return cost

        for cover in useful_covers:
            new_remaining = frozenset(remaining - cover)

            if new_remaining not in visited:
                visited.add(new_remaining)
                queue.append((new_remaining, cost + 1))

    return inf
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    relaxed_state = set(state)

    if goal.issubset(relaxed_state):
        return 0

    actions = get_all_groundings(domain, objects)
    steps = 0

    while not goal.issubset(relaxed_state):
        best_action = None
        best_goal_gain = -1
        best_total_gain = -1

        current_goal_count = len(set(goal) & relaxed_state)

        for action in actions:
            preconditions_ok = action.precond_pos.issubset(relaxed_state)
            negative_preconditions_ok = action.precond_neg.isdisjoint(relaxed_state)

            if not preconditions_ok or not negative_preconditions_ok:
                continue

            added_fluents = set(action.add_list) - relaxed_state

            if not added_fluents:
                continue

            next_state = relaxed_state | set(action.add_list)
            next_goal_count = len(set(goal) & next_state)

            goal_gain = next_goal_count - current_goal_count
            total_gain = len(added_fluents)

            if goal_gain > best_goal_gain:
                best_goal_gain = goal_gain
                best_total_gain = total_gain
                best_action = action
            elif goal_gain == best_goal_gain and total_gain > best_total_gain:
                best_total_gain = total_gain
                best_action = action

        if best_action is None:
            return inf

        before_size = len(relaxed_state)
        relaxed_state |= set(best_action.add_list)
        after_size = len(relaxed_state)

        if after_size == before_size:
            return inf

        steps += 1

    return steps
    ### End of your code ###
