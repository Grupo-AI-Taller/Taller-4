from __future__ import annotations

from collections import deque

from planning.pddl import Action, Problem, apply_action, is_applicable


# Infraestructura HTN


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# Punto 5a: hierarchicalSearch


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.
    """
    queue: deque = deque()
    queue.append(list(hlas[:1]))  # plan inicial: sólo la HLA raíz

    while queue:
        plan = queue.popleft()

        if is_plan_primitive(plan):
            state = problem.initial_state
            valid = True
            for action in plan:
                if not is_applicable(state, action):
                    valid = False
                    break
                state = apply_action(state, action)
            if valid and problem.isGoalState(state):
                return plan
        else:
            for i, step in enumerate(plan):
                if not is_primitive(step):
                    for refinement in step.refinements:
                        queue.append(plan[:i] + list(refinement) + plan[i + 1:])
                    break

    return []


# Punto 5b: definicion de HLAs y funciones auxiliares


def _bfs_path(start, end, adj: dict) -> list | None:
    """BFS sobre el grafo de adyacencia; retorna [start, ..., end] o None."""
    if start == end:
        return [start]
    frontier = deque()
    frontier.append((start, [start]))
    visited = {start}
    while frontier:
        current, path = frontier.popleft()
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                visited.add(neighbor)
                frontier.append((neighbor, new_path))
    return None


def _make_move(robot: str, a, b) -> Action:
    return Action(
        f"Move({robot},{a},{b})",
        [("At", robot, a), ("Adjacent", a, b), ("Free", b)],
        [],
        [("At", robot, b), ("Free", a)],
        [("At", robot, a), ("Free", b)],
    )


def _make_pickup(robot: str, obj: str, loc) -> Action:
    return Action(
        f"PickUp({robot},{obj},{loc})",
        [("At", robot, loc), ("At", obj, loc), ("HandsFree", robot), ("Pickable", obj)],
        [],
        [("Holding", robot, obj)],
        [("At", obj, loc), ("HandsFree", robot)],
    )


def _make_putdown(robot: str, obj: str, loc) -> Action:
    return Action(
        f"PutDown({robot},{obj},{loc})",
        [("At", robot, loc), ("Holding", robot, obj)],
        [],
        [("At", obj, loc), ("HandsFree", robot)],
        [("Holding", robot, obj)],
    )


def _make_setup(robot: str, supply: str, loc) -> Action:
    return Action(
        f"SetupSupplies({robot},{supply},{loc})",
        [("At", robot, loc), ("Holding", robot, supply), ("MedicalPost", loc)],
        [],
        [("SuppliesReady", loc), ("HandsFree", robot)],
        [("Holding", robot, supply)],
    )


def _make_rescue(robot: str, patient: str, loc) -> Action:
    return Action(
        f"Rescue({robot},{patient},{loc})",
        [("At", robot, loc), ("At", patient, loc), ("MedicalPost", loc), ("SuppliesReady", loc)],
        [],
        [("Rescued", patient)],
        [("At", patient, loc)],
    )


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    Returns a list where hlas[0] is the root HLA.
    """
    state = problem.initial_state
    objects = problem.objects
    robot = "robot"

    # extraer posiciones y adyacencia del estado inicial
    robot_pos = None
    supply_positions: dict[str, tuple] = {}
    patient_positions: dict[str, tuple] = {}
    medical_posts: list = []
    adj: dict = {}

    for fluent in state:
        pred = fluent[0]
        if pred == "At":
            _, entity, cell = fluent
            if entity == robot:
                robot_pos = cell
            elif entity in objects.get("supplies", []):
                supply_positions[entity] = cell
            elif entity in objects.get("patients", []):
                patient_positions[entity] = cell
        elif pred == "MedicalPost":
            medical_posts.append(fluent[1])
        elif pred == "Adjacent":
            a, b = fluent[1], fluent[2]
            adj.setdefault(a, []).append(b)

    if not medical_posts or robot_pos is None:
        return []

    med_post = medical_posts[0]

    # crear Navigate HLAs para cada par de celdas adyacentes
    navigate_hlas: dict[tuple, HLA] = {}
    for a, neighbors in adj.items():
        for b in neighbors:
            hla = HLA(f"Navigate({a},{b})", refinements=[[_make_move(robot, a, b)]])
            navigate_hlas[(a, b)] = hla

    def nav_sequence(path: list) -> list:
        """Convierte una lista de celdas [c0, c1, ..., cN] en Navigate HLAs consecutivos."""
        result = []
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key in navigate_hlas:
                result.append(navigate_hlas[key])
        return result

    supplies_list = objects.get("supplies", [])
    patients_list = objects.get("patients", [])

    current_robot = robot_pos  # posición esperada del robot al inicio de cada misión
    mission_hlas: list[HLA] = []

    for supply_name, patient_name in zip(supplies_list, patients_list):
        s_pos = supply_positions[supply_name]
        p_pos = patient_positions[patient_name]

        # PrepareSupplies: ir a los suministros, recogerlos y llevarlos al puesto medico
        path_to_supply = _bfs_path(current_robot, s_pos, adj) or [current_robot]
        path_supply_to_post = _bfs_path(s_pos, med_post, adj) or [s_pos]

        prepare_refinement = (
            nav_sequence(path_to_supply)
            + [_make_pickup(robot, supply_name, s_pos)]
            + nav_sequence(path_supply_to_post)
            + [_make_setup(robot, supply_name, med_post)]
        )
        prepare_hla = HLA(
            f"PrepareSupplies({supply_name},{med_post})",
            refinements=[prepare_refinement],
        )

        # ExtractPatient: el robot ya esta en med_post, va a buscar al paciente y lo trae
        path_to_patient = _bfs_path(med_post, p_pos, adj) or [med_post]
        path_patient_to_post = _bfs_path(p_pos, med_post, adj) or [p_pos]

        extract_refinement = (
            nav_sequence(path_to_patient)
            + [_make_pickup(robot, patient_name, p_pos)]
            + nav_sequence(path_patient_to_post)
            + [_make_putdown(robot, patient_name, med_post)]
            + [_make_rescue(robot, patient_name, med_post)]
        )
        extract_hla = HLA(
            f"ExtractPatient({patient_name},{med_post})",
            refinements=[extract_refinement],
        )

        full_mission_hla = HLA(
            f"FullRescueMission({supply_name},{patient_name},{med_post})",
            refinements=[[prepare_hla, extract_hla]],
        )
        mission_hlas.append(full_mission_hla)
        current_robot = med_post  # después de cada misión el robot queda en med_post

    if not mission_hlas:
        return []

    if len(mission_hlas) == 1:
        root_hla = mission_hlas[0]
    else:
        root_hla = HLA("MultiRescueMission", refinements=[mission_hlas])

    return [root_hla]
