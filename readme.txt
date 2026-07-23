venv\Scripts\activate

# Projet Agents IA — Azure OpenAI + AutoGen

## Structure
```
agents-project/
├── .env                    # tes vraies clés (à créer, jamais commité)
├── .env.example            # template
├── .gitignore
├── requirements.txt
├── config/
│   └── OAI_CONFIG_LIST.example
├── scripts/
│   └── test_connection.py  # valide la connexion Azure OpenAI
├── agents/
│   └── simple_agent.py     # premier agent AutoGen
└── logs/
```

## Installation

```bash
# 1. Créer l'environnement virtuel
python -m venv venv

# 2. Activer l'environnement virtuel
# Sur Mac/Linux :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les secrets
cp .env.example .env
# Puis édite .env avec ta vraie clé API et ton endpoint Azure

# 5. Tester la connexion
python scripts/test_connection.py

# 6. Lancer le premier agent AutoGen
python agents/simple_agent.py
```

## Notes
- Le nom de déploiement (`AZURE_OPENAI_DEPLOYMENT_NAME`) doit correspondre EXACTEMENT
  au nom donné lors du déploiement du modèle dans Azure AI Foundry (pas au nom du modèle lui-même).
- Ne jamais commit le fichier `.env` — il est déjà ignoré via `.gitignore`.