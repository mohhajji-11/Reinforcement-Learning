from env import FrozenLakeEnv, make_frozen_lake 
from agent import QLearningVisualAgent
import os
import pickle
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

# --- Fonctions utilitaires pour le chargement/création d'agent ---

def create_agent_from_params(env_params: Dict[str, Any], training_params: Optional[Dict[str, Any]] = None) -> QLearningVisualAgent:
    """Crée un nouvel agent avec les paramètres chargés ou par défaut."""
    
    # Gère la rétrocompatibilité : 'goals' (liste) ou 'goal' (tuple unique)
    goals_data = env_params.get('goals', env_params.get('goal'))
    
    env = FrozenLakeEnv(
        nrow=env_params['nrow'],
        ncol=env_params['ncol'],
        holes=env_params['holes'],
        goals=goals_data, 
        start_state=env_params['start_state']
    )
    
    # Paramètres d'entraînement par défaut ou chargés
    alpha = training_params.get('alpha', 0.1) if training_params else 0.1
    gamma = training_params.get('gamma', 0.9) if training_params else 0.9
    epsilon_min = training_params.get('epsilon_min', 0.01) if training_params else 0.01
    decay_rate = training_params.get('decay_rate', 0.9995) if training_params else 0.9995

    return QLearningVisualAgent(env, alpha=alpha, gamma=gamma, epsilon_min=epsilon_min, decay_rate=decay_rate)

def load_agent_and_env(filename: str) -> Optional[QLearningVisualAgent]:
    """Charge un modèle complet et crée l'environnement correspondant."""
    if not os.path.exists(filename):
        print(f"❌ Fichier {filename} non trouvé.")
        return None
    
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # 1. Créer l'agent avec les paramètres de l'environnement et d'entraînement
        env_params = model_data.get('env_params')
        if not env_params:
             print("❌ Les paramètres d'environnement (env_params) sont manquants dans le fichier de sauvegarde.")
             return None
             
        agent = create_agent_from_params(env_params, model_data.get('params'))
        
        # 2. Charger les données spécifiques (Q-table, historique, stats)
        if agent.load_model(filename):
            return agent
        else:
            return None
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement ou de la création de l'agent: {e}")
        return None

# --- Programme Principal ---

def main():
    print("🎬 Q-LEARNING FROZENLAKE AVEC VISUALISATION AVANCÉE")
    print("=" * 60)
    
    agent: Optional[QLearningVisualAgent] = None
    
    while True:
        print("\n--- MENU PRINCIPAL ---")
        if agent is None:
             print("1. 🛠️  Créer un Nouvel Agent (Environnement 5x5 par défaut)")
             print("2. 🧩 Créer un Nouvel Agent (Environnement Interactif)")
             print("3. 📂 Charger un Agent existant")
             print("4. 🚪 Quitter")
        else:
             num_goals = len(agent.env.goals)
             print(f"Agent actif: {agent.nrow}x{agent.ncol}, Goals: {num_goals}, Trous: {len(agent.env.holes)}, Épisodes Totaux: {agent.total_episodes}, ε: {agent.epsilon:.3f}")
             print("5. 🏋️ Entraîner l'Agent")
             print("6. 📊 Analyse détaillée des performances (Graphiques)")
             print("7. 🎯 Voir Politique Finale (Statique)")
             print("8. 💾 Sauvegarder l'Agent")
             print("9. 🚪 Quitter")

        choice = input("\nChoix : ").strip()
        
        # --- Création/Chargement ---
        if agent is None:
            if choice == "1":
                # Environnement par défaut (Goal unique par défaut)
                env = FrozenLakeEnv(nrow=5, ncol=5) 
                agent = QLearningVisualAgent(env)
            
            elif choice == "2":
                # Environnement interactif (utilise make_frozen_lake mis à jour)
                try:
                    env = make_frozen_lake(interactive=True)
                    env_params = {'nrow': env.nrow, 'ncol': env.ncol, 'holes': env.holes, 'goals': env.goals, 'start_state': env.start_state}
                    agent = create_agent_from_params(env_params)
                except ValueError as e:
                    print(f"❌ Erreur de configuration: {e}")
                
            elif choice == "3":
                filename = input("Nom du fichier à charger (ex: qlearning_model.pkl): ").strip() or "qlearning_model.pkl"
                agent = load_agent_and_env(filename)
            
            elif choice == "4":
                 print("👋 Au revoir!")
                 break
            
            else:
                 print("Choix invalide.")
                 
        # --- Actions sur l'Agent actif ---
        else: 
            if choice == "5":
                try:
                    episodes = int(input("Combien d'épisodes entraîner? (ex: 1000) "))
                    show_viz = input("Visualisation en direct (lente) ? (o/n): ").lower() == 'o'
                    agent.train(episodes=episodes, show_live_visualization=show_viz)
                except ValueError:
                    print("Entrée invalide pour les épisodes.")
            
            elif choice == "6":
                try:
                    window = int(input("Taille de la fenêtre pour la moyenne mobile (ex: 50): ") or 50)
                    agent.plot_detailed_analysis(window_size=window)
                except ValueError:
                    print("Entrée invalide.")
            
            elif choice == "7":
                agent._show_final_visualization()
                input("Appuyez sur Entrée pour continuer...")
            
            elif choice == "8":
                filename = input("Nom du fichier de sauvegarde: ").strip() or "qlearning_final.pkl"
                agent.save_model(filename)
            
            elif choice == "9":
                print("👋 Au revoir!")
                break
            
            else:
                 print("Choix invalide!")

if __name__ == "__main__":
    main()