import torch

def normalize_expression(X, size_factor, encoder_type):
    """Normalize gene expression data based on the specified encoder type.

    Args:
        X (torch.Tensor): Input gene expression matrix.
        size_factor (torch.Tensor): Size factors for normalization.
        encoder_type (str): Type of encoder for normalization. It can be one of the following:
                            - "proportions": Normalize by dividing by size factor.
                            - "log_gexp": Apply log transformation to gene expression data.
                            - "learnt_encoder": Apply log transformation to gene expression data.
                            - "learnt_autoencoder": Apply log transformation to gene expression data.
                            - "log_gexp_scaled": Apply log transformation after scaling by size factor.

    Returns:
        torch.Tensor: Normalized gene expression data.

    Raises:
        NotImplementedError: If the encoder type is not recognized.
    """
    if encoder_type == "proportions":
        X = X / size_factor
    elif encoder_type in ["log_gexp", "learnt_encoder", "learnt_autoencoder"]:
        X = torch.log1p(X)
    elif encoder_type == "log_gexp_scaled":
        X = torch.log1p(X / size_factor)
    else:
        raise NotImplementedError(f"Encoder type '{encoder_type}' is not implemented.")
    return X

def compute_size_factor_lognorm(adata, layer, id2cov):
    """Compute the mean and variance of the log size factors for each covariate category.

    Args:
        adata (AnnData): Annotated data matrix.
        layer (str): Name of the layer containing the gene expression data.
        id2cov (dict): Dictionary mapping covariate names to their categories.

    Returns:
        tuple: Two dictionaries containing the mean and standard deviation of the log size factors 
               for each covariate category.
               - log_size_factors_mean (dict): Mean log size factors per covariate category.
               - log_size_factors_sd (dict): Standard deviation of log size factors per covariate category.
    """
    log_size_factors_mean, log_size_factors_sd = {}, {}
    
    for cov_name in id2cov:
        log_size_factors_mean_cov, log_size_factors_sd_cov = [], []
        
        for cov_cat in id2cov[cov_name]:
            adata_cov = adata[adata.obs[cov_name] == cov_cat]
            log_size_factors_cov = torch.log(torch.tensor(adata_cov.layers[layer].todense().sum(1)))
            mean, sd = log_size_factors_cov.mean(), log_size_factors_cov.std()
            
            log_size_factors_mean_cov.append(mean)
            log_size_factors_sd_cov.append(sd)
        
        log_size_factors_mean[cov_name] = torch.stack(log_size_factors_mean_cov)
        log_size_factors_sd[cov_name] = torch.stack(log_size_factors_sd_cov)
    
    return log_size_factors_mean, log_size_factors_sd
