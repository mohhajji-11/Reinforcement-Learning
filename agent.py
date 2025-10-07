import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import os
import time
from typing import Tuple, List, Dict, Any, Optional

# --- Fonctions utilitaires ---
# Fonction utilitaire pour le lissage (Moyenne Mobile)
def moving_average(data: List[float], window_size: int) -> np.ndarray:
    """Calcule la moyenne mobile pour le lissage des courbes."""
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Simuler l'import de FrozenLakeEnv pour que le code soit autonome si besoin
try:
    from env import FrozenLakeEnv 
except ImportError:
    class FrozenLakeEnv:
        def __init__(self, nrow, ncol, holes, goals, start_state):
            self.nrow = nrow; self.ncol = ncol
            self.num_states = nrow * ncol; self.num_actions = 4
            self.desc = np.array([['F' for _ in range(ncol)] for _ in range(nrow)])
            self.start_state = start_state
            self.holes = holes; self.goals = goals
        def state_to_index(self, state): return state[0] * self.ncol + state[1]
        def index_to_state(self, index): return (index // self.ncol, index % self.ncol)
        def reset(self): return (self.start_state, {})
        def step(self, action): return ((0,0), 0.0, False, False, {})
        

class QLearningVisualAgent:
    """
    🎯 AGENT Q-LEARNING VISUEL POUR FROZENLAKE
    - Entraînement avec epsilon-decay
    - Visualisation de la politique et du chemin
    - Analyse statistique détaillée (convergence lissée, exploration vs. exploitation)
    """
    
    ACTION_NAMES = {0: "GAUCHE", 1: "BAS", 2: "DROITE", 3: "HAUT"}
    
    def __init__(self, env: FrozenLakeEnv, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.9995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        
        self.nrow, self.ncol = env.nrow, env.ncol
        self.n_states = env.num_states
        self.n_actions = env.num_actions
        
        # Initialisation de la Table Q
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # Historique détaillé pour l'analyse
        self.training_history: Dict[str, List[Any]] = {
            'rewards': [], 'success_rate': [], 'steps': [], 
            'epsilon': [], 'paths': []
        }
        
        self.total_episodes = 0
        self.best_reward = -np.inf
        self.best_path: List[Tuple[int, int]] = []
        
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.current_episode_path: List[Tuple[int, int]] = []
        
        print(f"🎯 Q-Learning Visual Agent créé pour FrozenLake {self.nrow}x{self.ncol}")
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        return self.env.state_to_index(state)
    
    def choose_action(self, state: Tuple[int, int], training=True) -> int:
        state_idx = self.state_to_index(state)
        
        if training and np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(self.env.action_space)
        else:
            # Exploitation
            return np.argmax(self.Q[state_idx])
    
    def decay_epsilon(self):
        """Décroissance exponentielle d'epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def train(self, episodes=1000, save_interval=100, display_interval=50, show_live_visualization=False):
        """Entraînement de l'agent."""
        
        if show_live_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(self.ncol + 1, self.nrow + 1))
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            state_idx = self.state_to_index(state)
            
            self.current_episode_path = [state]
            total_reward = 0
            done = False
            steps = 0
            success = False
            max_steps = self.n_states * 10 
            
            while not done and steps < max_steps:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state_idx = self.state_to_index(next_state)
                done = terminated or truncated
                
                # Mise à jour Q-Learning
                target = reward + self.gamma * np.max(self.Q[next_state_idx]) * (1 - terminated)
                self.Q[state_idx, action] += self.alpha * (target - self.Q[state_idx, action])
                
                total_reward += reward
                state_idx = next_state_idx
                state = next_state
                steps += 1
                
                self.current_episode_path.append(state)
                if terminated and reward > 0: success = True
                
                if show_live_visualization and (steps % 5 == 0 or done):
                    self._update_visualization(self.total_episodes + episode, total_reward, steps, success)
            
            # Mise à jour statistiques
            self.total_episodes += 1
            self.training_history['rewards'].append(total_reward)
            self.training_history['success_rate'].append(success)
            self.training_history['steps'].append(steps)
            self.training_history['epsilon'].append(self.epsilon)
            self.training_history['paths'].append(self.current_episode_path.copy())
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_path = self.current_episode_path.copy()
            
            self.decay_epsilon()
            
            if (episode + 1) % display_interval == 0:
                recent_rewards = self.training_history['rewards'][-display_interval:]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(self.training_history['success_rate'][-display_interval:])
                print(f"Ép. {self.total_episodes:4d} | Reward Moy: {avg_reward:6.2f} | Succès: {success_rate:5.1%} | ε: {self.epsilon:.3f}")
            
            if (episode + 1) % save_interval == 0:
                self.save_model("qlearning_checkpoint.pkl")
        
        if show_live_visualization and self.fig:
            plt.ioff()
            plt.close(self.fig)
        
        return self.training_history

    # --- Visualisation de Grille et Politique ---
    def _draw_grid_elements(self, ax: plt.Axes, current_path: Optional[List[Tuple[int, int]]] = None, current_pos: Optional[Tuple[int, int]] = None):
        """Dessine la grille, les objectifs, les obstacles et le chemin."""
        ax.clear()
        
        for r in range(self.nrow):
            for c in range(self.ncol):
                x, y_display = c, self.nrow - 1 - r 
                cell_type = self.env.desc[r, c]
                
                if cell_type == 'S': color, text, text_color = 'lightgreen', 'START', 'darkgreen'
                elif cell_type == 'F': color, text, text_color = 'lightblue', '', 'black'
                elif cell_type == 'H': color, text, text_color = 'red', 'HOLE', 'white'
                elif cell_type == 'G': 
                    color, text, text_color = 'gold', 'GOAL', 'darkred'
                    # Ajoute l'index si plusieurs goals existent
                    if len(self.env.goals) > 1:
                        try:
                           text = f'G{self.env.goals.index((r,c)) + 1}'
                        except ValueError:
                           text = 'GOAL' # Par sécurité
                        
                else: color, text, text_color = 'white', '', 'black'
                
                rect = patches.Rectangle((x, y_display), 1, 1, 
                                       facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
                ax.add_patch(rect)
                if text:
                    ax.text(x + 0.5, y_display + 0.5, text, ha='center', va='center', fontsize=10, color=text_color, fontweight='bold')

        # Chemin et Position de l'agent
        if current_path and len(current_path) > 1:
            path_x = [pos[1] + 0.5 for pos in current_path] 
            path_y = [self.nrow - pos[0] - 0.5 for pos in current_path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.6, label='Chemin')
            ax.scatter(path_x, path_y, c='green', s=50, marker='o', zorder=5)
        
        if current_pos:
            y_display = self.nrow - current_pos[0] - 0.5
            ax.scatter([current_pos[1] + 0.5], [y_display], c='yellow', s=150, marker='o', edgecolors='black', zorder=10)
        
        ax.set_xlim(0, self.ncol)
        ax.set_ylim(0, self.nrow)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_policy_arrows(self, ax: plt.Axes):
        """Dessine les flèches indiquant la meilleure action pour chaque état."""
        arrow_scale = 0.3
        
        for r in range(self.nrow):
            for c in range(self.ncol):
                state_idx = self.state_to_index((r, c))
                
                if self.env.desc[r, c] in ['H', 'G']: continue
                
                x, y_display = c + 0.5, self.nrow - r - 0.5 
                best_action = np.argmax(self.Q[state_idx])
                
                dr, dc = self.env.ACTIONS[best_action]
                dx, dy = dc * arrow_scale, -int(dr) * arrow_scale 
                
                ax.arrow(x, y_display, dx, dy, 
                         head_width=0.15, head_length=0.15, 
                         fc='purple', ec='purple', alpha=0.6, zorder=6)

    def _update_visualization(self, episode, reward, steps, success):
        """Met à jour la visualisation pendant l'entraînement."""
        if self.ax:
            self.ax.clear()
            self._draw_grid_elements(self.ax, current_path=self.current_episode_path, current_pos=self.current_episode_path[-1])
            self._draw_policy_arrows(self.ax)
            
            status = "🎯 SUCCÈS" if success else "⚡ EN COURS"
            self.ax.set_title(f"Q-Learning FrozenLake - Épisode {episode+1}\n"
                             f"Reward: {reward:.2f} | Steps: {steps} | {status}\n"
                             f"ε: {self.epsilon:.3f}", fontsize=10)
            
            plt.draw()
            plt.pause(0.01)
    
    def _show_final_visualization(self):
        """Affiche la politique finale et le meilleur chemin trouvé."""
        plt.ioff()
        fig, ax = plt.subplots(figsize=(self.ncol + 1, self.nrow + 1))
        self.env.reset()
        self._draw_grid_elements(ax, current_path=self.best_path)
        self._draw_policy_arrows(ax)
        ax.set_title(f"🎯 POLITIQUE FINALE - Q-Learning FrozenLake\n"
                     f"Épisodes: {self.total_episodes} | Meilleur Reward: {self.best_reward:.2f}", fontsize=12)
        plt.show(block=True)
    
    # --- Analyse Détaillée ---

    def plot_detailed_analysis(self, window_size=50):
        """
        Affiche les graphiques d'analyse de performance,
        incluant la convergence lissée et l'évolution d'epsilon.
        """
        if self.total_episodes < window_size:
            print(f"❌ Entraînez l'agent pendant au moins {window_size} épisodes pour une analyse lissée significative.")
            return

        plt.ioff()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        episodes = np.arange(1, self.total_episodes + 1)
        
        # Lissage des données
        returns = np.array(self.training_history['rewards'])
        steps_raw = np.array(self.training_history['steps'])
        success_rate_raw = np.array(self.training_history['success_rate'], dtype=float)
        
        smooth_returns = moving_average(returns, window_size)
        smooth_steps = moving_average(steps_raw, window_size)
        smooth_success = moving_average(success_rate_raw, window_size)
        episodes_ma = episodes[window_size - 1:]

        # Graphique 1: Décroissance d'Epsilon 
        ax = axes[0]
        ax.plot(episodes, self.training_history['epsilon'], color='mediumblue', linewidth=2)
        ax.set_title("1. Décroissance d'Epsilon (Exploration)")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Epsilon ($\epsilon$)")
        ax.grid(True, alpha=0.3)

        # Graphique 2: Retours cumulés par épisode (brut)
        ax = axes[1]
        ax.plot(episodes, returns, color='lightgray', linewidth=0.5, label='Retour brut')
        ax.set_title("2. Retours cumulés par épisode (brut)")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Retour cumulé (brut)")
        ax.grid(True, alpha=0.3)

        # Graphique 3: Convergence lissée 
        ax = axes[2]
        ax.plot(episodes_ma, smooth_returns, label=f"Moy. Mobile (Fenêtre={window_size})", color='darkgreen', linewidth=2)
        ax.set_title(f"3. Convergence lissée (fenêtre={window_size})")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Retour cumulé (moyenne mobile)")
        ax.grid(True, alpha=0.3)
        
        # Graphique 4: Longueur des épisodes (lissée)
        ax = axes[3]
        ax.plot(episodes_ma, smooth_steps, color='indigo', linewidth=2)
        ax.set_title("4. Longueur des épisodes (lissée)")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Nombre de pas")
        ax.grid(True, alpha=0.3)

        # Graphique 5: Taux de Succès lissé
        ax = axes[4]
        ax.plot(episodes_ma, smooth_success, color='darkorange', linewidth=2)
        ax.set_title(f"5. Taux de Succès (Lissé, Fenêtre={window_size})")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Taux de Succès (%)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Graphique 6: Exploration vs Exploitation
        ax = axes[5]
        epsilon_values = np.array(self.training_history['epsilon'])
        exploration = epsilon_values
        
        ax.plot(episodes, exploration, color='orange', label='Exploration', linewidth=0.5)
        ax.plot(episodes, 1 - exploration, color='blue', label='Exploitation', linewidth=0.5)
        
        ax.fill_between(episodes, 0, exploration, color='orange', alpha=0.5, label='Exploration')
        ax.fill_between(episodes, exploration, 1, color='blue', alpha=0.5, label='Exploitation')
        
        ax.set_title("6. Exploration vs Exploitation")
        ax.set_xlabel("Épisodes")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Analyse Détaillée Q-Learning sur FrozenLake {self.nrow}x{self.ncol}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=True)
        
    # --- Sauvegarde et Chargement ---
    
    def save_model(self, filename: str = "qlearning_model.pkl"):
        """Sauvegarde la table Q et les données de l'environnement."""
        env_params = {
            'nrow': self.nrow, 'ncol': self.ncol, 'holes': self.env.holes, 
            'goals': self.env.goals, # Sauvegarde la liste de goals
            'start_state': self.env.start_state,
        }
        
        model_data = {
            'Q_table': self.Q,
            'training_history': self.training_history,
            'params': {'alpha': self.alpha, 'gamma': self.gamma, 'epsilon_min': self.epsilon_min, 'decay_rate': self.decay_rate},
            'statistics': {'total_episodes': self.total_episodes, 'best_reward': self.best_reward, 'epsilon': self.epsilon},
            'best_path': self.best_path,
            'env_params': env_params
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modèle sauvegardé: {filename}")
    
    def load_model(self, filename: str = "qlearning_model.pkl") -> bool:
        """Charge la table Q, l'historique et les paramètres."""
        if not os.path.exists(filename):
            print(f"❌ Fichier {filename} non trouvé!")
            return False
        
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)

            loaded_Q = model_data['Q_table']
            if loaded_Q.shape != self.Q.shape:
                 print(f"❌ Erreur: Le modèle chargé ({loaded_Q.shape}) est incompatible avec la grille actuelle ({self.Q.shape}).")
                 return False

            self.Q = loaded_Q
            self.training_history = model_data.get('training_history', self.training_history)
            
            # Mise à jour des hyperparamètres et statistiques
            self.alpha = model_data['params'].get('alpha', self.alpha)
            self.gamma = model_data['params'].get('gamma', self.gamma)
            self.epsilon_min = model_data['params'].get('epsilon_min', self.epsilon_min)
            self.decay_rate = model_data['params'].get('decay_rate', self.decay_rate)
            self.epsilon = model_data['statistics']['epsilon'] 
            self.total_episodes = model_data['statistics']['total_episodes']
            self.best_reward = model_data['statistics']['best_reward']
            self.best_path = model_data.get('best_path', [])
            
            print(f"📂 Modèle chargé: {filename}. Reprend à ε={self.epsilon:.3f}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du fichier: {e}")
            return False

if __name__ == "__main__":
    pass