import wandb

sweep_config = {
    'method': 'grid',
    'name':   'sweep',
    'project': 'notebook_example',
    'metric': {
        'goal': 'minimize',
        'name': 'average'
    },
    'parameters': {
        'value1': {'values': [2, 3, 4]},
        'value2': {'values': [4, 3, 2]},
        'method': {'values': ['sum', 'prod']},
        'lonely': {'value': 3}
    }
}

def training():
    wandb.init()
    v1 = wandb.config.value1
    v2 = wandb.config.value2
    mtd = wandb.config.method
    lnl = wandb.config.lonely

    for i in range(lnl):
        if mtd == 'sum':
            avg = (v1 + v2 + i) / 3
        else:
            avg = (v1 * v2 * i) / 3
            
        wandb.log({"value1": v1, "value2": v2, "method": mtd, "average": avg})

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=training)