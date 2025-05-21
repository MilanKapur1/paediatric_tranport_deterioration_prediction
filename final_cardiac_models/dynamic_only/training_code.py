

transformer_encoder = TransformerEncoderLayer(
    dim_in=18,      # Input feature dimension before concatenation
    dim_out=128,    # Output dimension
    max_seq_len=924,
    depth=4,
    heads=2
)

# Define model parameters
combined_mlp_hidden_dim = 512
output_dim = 1  # For two outputs: respiratory and cardiac deterioration probabilities

# Instantiate the combined model
model = CombinedModel(
    transformer_encoder=transformer_encoder,
    combined_mlp_hidden_dim=combined_mlp_hidden_dim,
    output_dim=output_dim
)
model = torch.compile(model)




pos_weight = torch.tensor(30.0, device=device)  # Adjust this value based on imbalance ratio

# Regularization parameters
l1_lambda = 0  # Adjust as necessary for L1 regularization
l2_lambda = 0  # For L2 regularization (weight decay)

device = torch.device("cpu")

# Initialize optimizer with only trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(
    trainable_params,
    lr=1e-5,
    weight_decay=l2_lambda  # Set your L2 regularization coefficient here
)


trained_model = fit(
    model=model,
    experiment_name='exp262',
    num_epochs=20,
    optimizer=optimizer,
    pos_weight=pos_weight,
    l1_lambda=l1_lambda,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    max_norm=10,
    lr_patience=5,
    patience=20,
    target_outcome = 'cardiac'
)