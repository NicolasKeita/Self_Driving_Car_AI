import gymnasium as gym
import typer

from modules.dqn import DQNAgent
from modules.utils import plot_results

app = typer.Typer()

# Set default values for n_episodes and render
DEFAULT_N_EPISODES = 1000
DEFAULT_RENDER = True

@app.command()
def main(n_episodes: int = DEFAULT_N_EPISODES, render: bool = DEFAULT_RENDER) -> None:
    if render:
        env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
    else:
        env = gym.make("CarRacing-v2", continuous=False, render_mode=None)

    agent = DQNAgent(env, gamma=0.99, epsilon_init=1.0, epsilon_min=0.05, epsilon_decay=0.7)
    agent.load_model("models", "dqn_final")
    results = agent.play(n_episodes)

    scores = results["score"]
    avg_score = sum(scores) / len(scores)
    print("---------------------")
    print(f"Average score: {avg_score:.2f}")
    print("---------------------")
    input("Press ENTER to exit.")


if __name__ == "__main__":
    app()
