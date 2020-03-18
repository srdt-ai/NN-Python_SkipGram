ef initialize_wrd_emb(vocab_size, emb_size):
    """
    vocab_size: int. taille du vocabulaire du corps/ensemble d'entrainement
    emb_size: int. taille des embedded words. Det dimensions pour representer le vocabulaire
    """
    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    return WRD_EMB

def initialize_dense(input_size, output_size):
    """
    input_size: int. taille de l'entree de la couche 'dense'
    output_szie: int. taille de la sortie de la couche 'dense'
    """
    W = np.random.randn(output_size, input_size) * 0.01
    return W

def initialize_parameters(vocab_size, emb_size):
    """
    initialise tous parametres d entrainement
    """
    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)
    
    parameters = {}
    parameters['WRD_EMB'] = WRD_EMB
    parameters['W'] = W
    
    return parameters

    