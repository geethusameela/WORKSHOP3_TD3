import json

# Initialize model balances
model_balances = {}

# Function to register a model and make an initial deposit
def register_model(model_id, initial_deposit):
    model_balances[model_id] = {'balance': initial_deposit}

# Function to penalize a model (slashing)
def penalize_model(model_id, penalty_amount):
    model_balances[model_id]['balance'] -= penalty_amount

# Function to reward a model
def reward_model(model_id, reward_amount):
    model_balances[model_id]['balance'] += reward_amount

# Function to adjust model weights based on balances
def adjust_weights():
    total_balance = sum(model['balance'] for model in model_balances.values())
    for model_id, model_info in model_balances.items():
        model_weights[model_id] = model_info['balance'] / total_balance


register_model('model1', 1000)
register_model('model2', 1000)

# Penalize model1 for inaccurate predictions
penalize_model('model1', 100)

# Reward model2 for accurate predictions
reward_model('model2', 200)

# Adjust weights based on balances
adjust_weights()

# Save model balances to JSON file
with open('model_balances.json', 'w') as f:
    json.dump(model_balances, f)
