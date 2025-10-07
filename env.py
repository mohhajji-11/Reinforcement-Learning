import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Any, List, Optional, Union

# Le Goal peut être une position unique ou une liste de positions
GoalType = Union[Tuple[int, int], List[Tuple[int, int]]] 

class FrozenLakeEnv:
    """
    Un environnement FrozenLake flexible supportant plusieurs buts (Goals).
    S: Départ, F: Gelé, H: Trou, G: But
    """
    # Actions: 0: GAUCHE, 1: BAS, 2: DROITE, 3: HAUT
    ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
    ACTION_NAMES = {0: "GAUCHE", 1: "BAS", 2: "DROITE", 3: "HAUT"}

    def __init__(self, nrow: int = 5, ncol: int = 5, holes: Optional[List[Tuple[int, int]]] = None, 
                 goals: Optional[GoalType] = None, start_state: Tuple[int, int] = (0, 0)):
        
        self.nrow = nrow
        self.ncol = ncol
        self.num_states = self.nrow * self.ncol
        self.num_actions = 4
        
        self.desc = np.array([['F' for _ in range(ncol)] for _ in range(nrow)])
        
        self.start_state = start_state
        self.holes = holes if holes is not None else []
        
        # Gère goals comme une liste
        if goals is None:
            self.goals = [(nrow - 1, ncol - 1)]
        elif isinstance(goals, tuple):
            self.goals = [goals]
        else:
            self.goals = goals
        
        # Validation et placement des éléments S, H, G
        self._set_grid_elements()
        
        self.state = self.start_state
        # Les états terminaux incluent tous les goals et tous les trous
        self.terminal_states = self.holes + self.goals
        self.action_space = list(self.ACTIONS.keys())

    def _set_grid_elements(self):
        """Place S, H, G dans la grille avec validation."""
        
        # La liste de toutes les positions occupées (pour vérifier les chevauchements)
        all_elements = {self.start_state}
        
        # 1. Validation du Départ
        if not (0 <= self.start_state[0] < self.nrow and 0 <= self.start_state[1] < self.ncol):
             raise ValueError(f"Start state {self.start_state} out of bounds.")

        # 2. Placement et validation des Goals
        for r, c in self.goals:
            if not (0 <= r < self.nrow and 0 <= c < self.ncol):
                 raise ValueError(f"Goal position {(r, c)} out of bounds.")
            if (r, c) in all_elements:
                raise ValueError(f"Goal position {(r, c)} conflicts with start or another goal.")
            all_elements.add((r, c))
            self.desc[r, c] = 'G'
            
        # 3. Placement et validation des Trous (Holes)
        for r, c in self.holes:
            if not (0 <= r < self.nrow and 0 <= c < self.ncol):
                 raise ValueError(f"Hole position {(r, c)} out of bounds.")
            if (r, c) in all_elements:
                raise ValueError(f"Hole position {(r, c)} conflicts with start or a goal.")
            all_elements.add((r, c))
            self.desc[r, c] = 'H'

        # 4. Placement du Départ
        self.desc[self.start_state] = 'S'


    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Réinitialise l'environnement (Gymnasium API)."""
        self.state = self.start_state
        return self.state, {}

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict[str, Any]]:
        """Exécute l'action (Gymnasium API)."""
        
        r, c = self.state
        dr, dc = self.ACTIONS[action]
        next_r, next_c = r + dr, c + dc
        
        # Gestion des limites
        next_r = max(0, min(next_r, self.nrow - 1))
        next_c = max(0, min(next_c, self.ncol - 1))

        next_state = (next_r, next_c)
        self.state = next_state
        
        terminated = next_state in self.terminal_states
        truncated = False
        reward = -0.01 
        
        cell_type = self.desc[next_r, next_c]

        if cell_type == 'G': 
            reward = 1.0 
        elif cell_type == 'H':
            reward = -1.0 

        info = {}
        return next_state, reward, terminated, truncated, info

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convertit l'état (r, c) en index unidimensionnel."""
        return state[0] * self.ncol + state[1]

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convertit l'index unidimensionnel en état (r, c)."""
        return (index // self.ncol, index % self.ncol)
    
    def render(self, mode: str = 'matplotlib'):
        pass 

# --- Fonctions pour l'interaction utilisateur ---

def get_coordinates_list(prompt_base: str, count: int, nrow: int, ncol: int, exclude_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Fonction utilitaire pour demander des coordonnées multiples."""
    coords_list = []
    
    for i in range(count):
        while True:
            try:
                coord_input = input(f"➡️ {prompt_base} #{i+1}/{count} ('row,col'): ").strip()
                if not coord_input: 
                     print("❌ Veuillez entrer des coordonnées.")
                     continue
                
                pos = tuple(map(int, coord_input.split(',')))
                
                if not (0 <= pos[0] < nrow and 0 <= pos[1] < ncol):
                    print("❌ Coordonnées hors limites.")
                    continue
                if pos in exclude_list:
                    print(f"❌ Ces coordonnées chevauchent une position déjà définie (Départ, Trou ou Goal).")
                    continue
                if pos in coords_list:
                    print("❌ Cette position est déjà définie pour cet élément.")
                    continue
                
                coords_list.append(pos)
                exclude_list.append(pos) # Ajouter à la liste d'exclusion pour les prochains inputs
                print(f"✅ Ajouté {prompt_base.lower()} à {pos}")
                break
                
            except ValueError:
                print("❌ Format invalide. Entrez 'row,col'.")
    return coords_list

def make_frozen_lake(nrow: int = 5, ncol: int = 5, holes: Optional[List[Tuple[int, int]]] = None, 
                     goals: Optional[GoalType] = None, start_state: Tuple[int, int] = (0, 0), 
                     interactive: bool = False) -> FrozenLakeEnv:
    """Crée l'environnement FrozenLake. Si interactive=True, demande les paramètres à l'utilisateur."""
    
    if not interactive:
        return FrozenLakeEnv(nrow=nrow, ncol=ncol, holes=holes, goals=goals, start_state=start_state)

    print("\n🏒 Environnement FrozenLake Interactif 🏒")
    
    # Liste des positions occupées (exclues des inputs suivants)
    occupied_positions: List[Tuple[int, int]] = []
    
    # 1. DIMENSIONS
    while True:
        try:
            input_dim = input("➡️ Entrez les dimensions 'Lignes,Colonnes' (ex: 5,5, min 2x2): ").strip()
            nrow, ncol = map(int, input_dim.split(','))
            if nrow >= 2 and ncol >= 2: break
            print("❌ Les dimensions doivent être d'au moins 2x2.")
        except ValueError:
            print("❌ Format invalide. Entrez 'Ligne,Colonne'.")

    # 2. POSITION DE DÉPART (Start)
    while True:
        try:
            start_input = input(f"➡️ Position de départ 'row,col' (0-{nrow-1}, 0-{ncol-1}, défaut 0,0): ").strip()
            start_state = (0, 0) if not start_input else tuple(map(int, start_input.split(',')))
            
            if 0 <= start_state[0] < nrow and 0 <= start_state[1] < ncol:
                occupied_positions.append(start_state)
                break
            print("❌ Position de départ hors limites.")
        except ValueError:
            print("❌ Format invalide. Entrez 'row,col'.")
    
    # 3. GOALS (Buts)
    
    max_goals = nrow * ncol - 1 
    while True:
        try:
            default_goal_count = 1 if nrow * ncol > 1 else 0
            num_goals_input = input(f"\n➡️ Combien de Goals (Buts) voulez-vous? (1 à {max_goals}, défaut 1): ").strip()
            num_goals = int(num_goals_input) if num_goals_input else default_goal_count
            
            if 1 <= num_goals <= max_goals:
                break
            print(f"❌ Nombre invalide. Le nombre de buts doit être entre 1 et {max_goals}.")
        except ValueError:
            print("❌ Entrée invalide. Entrez un nombre entier.")
    
    print("\n🥅 Saisie des coordonnées des Goals:")
    goals_list = get_coordinates_list("Goal", num_goals, nrow, ncol, occupied_positions)
    
    # 4. OBSTACLES (Trous)
    
    max_holes = nrow * ncol - len(occupied_positions)
    while True:
        try:
            num_holes_input = input(f"\n➡️ Combien de Trous (Obstacles) voulez-vous? (0 à {max_holes}): ").strip()
            num_holes = int(num_holes_input)
            if 0 <= num_holes <= max_holes:
                break
            print(f"❌ Nombre invalide. Le nombre de trous doit être entre 0 et {max_holes}.")
        except ValueError:
            print("❌ Entrée invalide. Entrez un nombre entier.")
    
    print("\n🕳️  Saisie des coordonnées des Trous:")
    # La fonction gère la mise à jour de occupied_positions
    holes_list = get_coordinates_list("Trou", num_holes, nrow, ncol, occupied_positions)
            
    return FrozenLakeEnv(nrow=nrow, ncol=ncol, holes=holes_list, goals=goals_list, start_state=start_state)

if __name__ == "__main__":
    pass