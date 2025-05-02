from models import Encoder, Classifier, Discriminator

def build_network(config):
    seq_len = config['model']['seq_len']
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    e_node = config['model']['e_node']
    c_node = config['model']['c_node']
    out_node = config['model']['out_node']
    strategy = config['model']['strategy']
    bias = config['model']['bias']

    encoder = Encoder(seq_len, input_dim, hidden_dim, e_node, strategy, bias)
    classifier = Classifier(hidden_dim, c_node, out_node)

    if config['train']['method'].lower() in ['dann', 'adda']:
        discriminator = Discriminator(hidden_dim, config['model']['d_node'])
        net = dotdict({
            'encoder': encoder,
            'classifier': classifier,
            'discriminator': discriminator,
        })
    elif config['train']['method'].lower() in ['mcd', 'hhd']:
        second_classifier = Classifier(hidden_dim, c_node, out_node)
        net = dotdict({
            'encoder': encoder,
            'classifier': classifier,
            'second_classifier': second_classifier,
        })
    else:
        net = dotdict({
            'encoder': encoder,
            'classifier': classifier,
        })
    return net


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]