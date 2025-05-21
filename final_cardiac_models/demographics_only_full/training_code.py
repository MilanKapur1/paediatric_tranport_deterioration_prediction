

batch_size = 64


scalar_input_dim = 140  # Example static feature dimension
embedding_input_dim = 768 # Example static feature dimension
scalar_mlp_hidden_dim = 256
embedding_hidden_dim = 256
combined_mlp_hidden_dim = 512
output_dim = 1  # For two outputs: respiratory and cardiac deterioration probabilities

# Instantiate the combined model
model = CombinedModel(
    scalar_input_dim=scalar_input_dim,
    embedding_input_dim=embedding_input_dim,
    scalar_mlp_hidden_dim=scalar_mlp_hidden_dim,
    embedding_hidden_dim=embedding_hidden_dim,
    combined_mlp_hidden_dim=combined_mlp_hidden_dim,
    output_dim=output_dim
)
model = torch.compile(model)




pos_weight = torch.tensor(20.0, device=device)  # Adjust this value based on imbalance ratio

# Regularization parameters
l1_lambda = 0  # Adjust as necessary for L1 regularization
l2_lambda = 0.003  # For L2 regularization (weight decay)

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
    experiment_name='demographics_only_full',
    num_epochs=25,
    optimizer=optimizer,
    pos_weight=pos_weight,
    l1_lambda=l1_lambda,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    max_norm=10,
    lr_patience=5,
    patience=20
)