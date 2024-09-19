import torch
import torch.nn as nn
import torch.optim as optim

def polynomial_decay(a, b, decay_steps, end_learning_rate=2e-7, power=1.0):
    return end_learning_rate + (a - end_learning_rate) * ((1 - b / decay_steps) ** power)

def lstm(network_input, state_ph, dropout_ph, early_stop_ph,
         n_layers, state_size, n_labels, cell_type='gru'):
    if cell_type == 'gru':
        rnn_cell = nn.GRU
    elif cell_type == 'lstm':
        rnn_cell = nn.LSTM
    else:
        raise ValueError("Unsupported cell type")

    batch_size = network_input.size(0)
    sequence_length = network_input.size(1)

    # Prepare initial state
    h_0 = state_ph.view(n_layers, batch_size, state_size)
    if cell_type == 'lstm':
        c_0 = torch.zeros_like(h_0)
        initial_state = (h_0, c_0)
    else:
        initial_state = h_0

    # Define RNN cell
    rnn = rnn_cell(input_size=network_input.size(-1), hidden_size=state_size, num_layers=n_layers, dropout=dropout_ph)

    # Forward pass
    states_series, current_state = rnn(network_input, initial_state)

    # Extract the last state
    last = states_series[torch.arange(batch_size), early_stop_ph - 1, :]

    # Output layer
    logits = nn.Linear(state_size, n_labels)(last)

    return logits, current_state

def create_summary(variables, types, names):
    """
    variables: a list of tensor variables
    types: a list strings of either 'scalar','image','histogram' of same length as variables
    names: a list of strings for the names of each summary
    """
    summary_ops = []
    for variable, type, name in zip(variables, types, names):
        if type == 'scalar':
            summary_ops.append(nn.functional.scalar_summary(name, variable))
        elif type == 'image':
            summary_ops.append(nn.functional.image_summary(name, variable, max_outputs=2))
        elif type == 'histogram':
            summary_ops.append(nn.functional.histogram_summary(name, variable))
        else:
            raise ValueError("Not valid summary type")
    return summary_ops

def create_solver(loss, global_step, learning_rate_ph, decay_step_ph, params, increment_global_step, optimizer='Adam', **kwargs):
    optimizer_class = getattr(optim, optimizer)
    optimizer_instance = optimizer_class(params, lr=learning_rate_ph, **kwargs)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer_instance, lr_lambda=lambda b: polynomial_decay(
        learning_rate_ph, b, decay_steps=decay_step_ph, **kwargs))

    return optimizer_instance, scheduler
