This file shows the Colab-based HF TRL + LoRA pipeline used to train and evaluate LLM behavior in EnergyMind.

We generate supervised examples directly from the environment. Each row contains an EnergyMind state and the target action/strategy output. In this run, we generated 14,400 compact environment-grounded examples.


<img width="1913" height="913" alt="hftrl-4" src="https://github.com/user-attachments/assets/f88e8fc1-6db9-44af-b9ff-c56be7db63ff" />

Direct room-level LLM control turned out to be the wrong abstraction. The base model produced invalid or weak outputs, and even the fine-tuned model often collapsed to trivial all-zero actions.


<img width="1915" height="913" alt="hftrl-3" src="https://github.com/user-attachments/assets/343eaf2c-c594-48ec-9e5e-d94f8b37e4dd" />

We fine-tune a base model in Colab using Hugging Face TRL and LoRA. Instead of retraining the full model, we train a lightweight adapter on environment-generated trajectories, which makes the pipeline practical and reproducible.


<img width="1905" height="917" alt="hftrl-5" src="https://github.com/user-attachments/assets/b37aaeda-25fb-4643-80e4-18c0fd1fa81a" />


