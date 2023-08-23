import numpy as np


def generate_hierarchical_data(alpha, beta, n_sample, proportion=None):
    # Parameters
    n_class = len(alpha.keys())
    n_pathway = len(beta.keys())
    n_gene = len(beta[list(beta.keys())[0]])
    n_read = 1500000
    
    # Generate data
    X = np.zeros((n_sample, n_gene))
    y = np.zeros((n_sample), dtype='int')

    count = 0
    for c in range(n_class):
        # Number of samples per class
        if proportion is None:
            n_sample_per_class = int(n_sample/n_class)
        else:
            n_sample_per_class = int(proportion[c] * n_sample)
        genes = np.zeros((n_sample_per_class, n_gene))
        # Labels (n_sample)
        y[count:count+n_sample_per_class] = np.ones((n_sample_per_class), dtype='int') * c
        # Proportion of pathways in a sample (n_sample, n_pathway)
        class_to_path = np.zeros((n_sample_per_class, n_pathway))
        values = np.random.dirichlet(alpha['C'+str(c)][alpha['C'+str(c)]!=0], size=(n_sample_per_class))
        class_to_path[:, alpha['C'+str(c)]!=0] = values
        # Each sample contains n_read reads associated with genes.
        for sample in range(n_sample_per_class):
            print(count+sample, end='\r')
            # For each read, draw a pathway
            pathway = np.random.multinomial(n_read, class_to_path[sample])
            # For each read, draw a gene
            for p in range(n_pathway):
                reads = np.random.multinomial(pathway[p], beta['P'+str(p)])
                genes[sample] += reads
        genes /= (n_read/1000000)    
        X[count:count+n_sample_per_class, :] = genes
        count += n_sample_per_class
    return X, y


def generate_eta_beta(n_pathway, sparsity, n_gene, case=None):
    eta = {}
    beta = {}
    for p in range(n_pathway):
        # Define the underlying graph structure
        if case != 0:
            if sparsity not in ["overlap", "overlap2"]:
                eta['P'+str(p)] = np.array([0.,] * case * p + [1.,] * case + [0.,] * case * (n_pathway - p - 1)) * 5
            else:
                eta['P'+str(p)] = np.array([0.,] * n_gene)
                if sparsity == "overlap" and p == n_pathway - 1:
                    eta['P'+str(p)][(case-1)*p:(case-1)*p+case-1] = np.array([1.,] * (case-1)) * 5
                    eta['P'+str(p)][0] = np.array([1.,]) * 5
                else:
                    eta['P'+str(p)][(case-1)*p:(case-1)*p+case] = np.array([1.,] * case) * 5
        else:
            eta['P'+str(p)] = np.random.binomial(1, sparsity, size=[n_gene]) * 5.
            while np.sum(eta['P'+str(p)]) < 3:
                eta['P'+str(p)] = np.random.binomial(1, sparsity, size=[n_gene]) * 5.
        # Attribute weights to each edge
        beta['P'+str(p)] = np.zeros((n_gene))
        values = np.random.dirichlet(eta['P'+str(p)][eta['P'+str(p)]!=0], size=(1))
        beta['P'+str(p)][eta['P'+str(p)]!=0] = values.reshape(-1)
    return eta, beta


def return_parameters(name):
    # Number of classes and number of variables
    if name in ['SIMU1', 'SIMU2']:
        n_class = 33
        n_gene = 15000
    elif name in ['SimuB']:
        n_class = 4
        n_gene = 50000
    elif name in ['SimuA',]:
        n_class = 2
        n_gene = 50000
    elif name in ['SimuC',]:
        n_class = 6
        n_gene = 50000
    elif name.split("_")[0] == "syn":
        n_gene = 2000
        if name.split("_")[1] == "g":
            n_class = int(name.split("_")[2]) + 1
        else:
            n_class = 2
    elif name == "demo":
        n_class = 2
        n_gene = 60
    elif name in ["set0", "set2", "set3"]:
        n_class = 3
        n_gene = 6
    elif name == "set1":
        n_class = 3
        n_gene = 26
    elif name == "set4":
        n_class = 3
        n_gene = 3
    elif name == "set5":
        n_class = 3
        n_gene = 4
    
    # Distribution of the examples among the classes
    if name.split("_")[0] == "syn" and name.split("_")[1] == "g" and name.split("_")[2] in ["5", "10"]:
        if name.split("_")[2] == "5":
            proportion = [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]
        elif name.split("_")[2] == "10":
            proportion = [1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/2]
    else:
        proportion = None  # if None, generate a balanced number of samples per class
    
    alpha = {}
    
    # Number of pathways
    if name == 'SIMU1':
        n_pathway = 1500
    elif name == 'SIMU2':
        n_pathway = 3000
    elif name in ['SimuA', 'SimuB', 'SimuC']:
        n_pathway = 5000
    elif name.split("_")[0] == "syn":
        if name.split("_")[1] in ["g", "p"]:
            n_pathway = 200
        elif name.split("_")[1] == "t":
            n_pathway = int(2000 / int(name.split("_")[2]))
    elif name == "demo":
        n_pathway = 6
    elif name in ["set0", "set2", "set3", "set4", "set5"]:
        n_pathway = 3
    elif name == "set1":
        n_pathway = 13
    for c in range(n_class):
        alpha['C' + str(c)] = np.array([1.] * n_pathway)  # each pathway has a priori the same importance
   # else:
   #     for c in range(n_class):
   #         alpha['C' + str(c)] = np.array([0.] * n_pathway)  # each pathway has a priori no importance


    # Number of overexpressed pathways
    useful_paths = []
    cls_gap = 2.
    if name in ['SIMU1', 'SIMU2']:
        P = 37
    elif name in ['SimuA', 'SimuB', 'SimuC']:
        P = 500
    elif name.split("_")[0] == "syn":
        if name.split("_")[1] == "g":
            P = 10
        elif name.split("_")[1] == "t":
            P = 1
        elif name.split("_")[1] == "p":
            P = int(name.split("_")[2])
    elif name == "demo":
        P = 2
    elif name in ["set0", "set1", "set2", "set3", "set4", "set5"]:
        P = 1
            
    useful_paths = {}
    for c in range(n_class):
        useful_paths['C' + str(c)] = []                
        for p in range(P):
            alpha['C' + str(c)][P * c + p] = cls_gap
            useful_paths['C' + str(c)].append('P' + str(P * c + p))
            
    # Variance
    for c in range(n_class):
        alpha['C'+str(c)] = alpha['C'+str(c)] * 4.
        
    # Prior on gene distribution per pathway   
    if name in ['SIMU1', "SimuA", "SimuB", "SimuC", "demo"]:
        case = 10
        sparsity = None
    elif name in ['SIMU2', ]:
        case = 0
        sparsity = 1 / n_gene * 10
    elif name.split("_")[0] == "syn":
        if name.split("_")[1] == "t":
            case = int(name.split("_")[2])
        else:
            case = 10
        sparsity = None
    elif name in [ "set0", "set1", "set2", "set3"]:
        case = 2
        sparsity = None
    elif name == "set4":
        case = 2
        sparsity = "overlap"
    elif name == "set5":
        case = 2
        sparsity = "overlap2"

    # Drawn gene distribution per pathway
    eta, beta = generate_eta_beta(n_pathway, sparsity, n_gene, case=case)

    # Store important genes
    useful_genes = {}
    for c in range(n_class):
        for P in (useful_paths["C" + str(c)]):
            useful_genes[P] = np.argwhere(eta[P] != 0).reshape(-1)

    # Check validity (useful genes must have a drawing probability relatively high)
    for P in useful_genes.keys():
        validity = False
        if case != 0:
            min_prop = 0.1 / case
            print(f"Group {P} with {case} variables: drawing probability >= {min_prop}.")
        else:
            min_prop = np.round(0.1 / (sparsity * n_gene), 2)
            print(f"Group {P} with around {np.round(sparsity * n_gene, 2)} variables: drawing probability >= {min_prop}.")
        while not validity:
            if min(beta[P][useful_genes[P]]) >= min_prop:
                    validity = True
            else:
                temp_eta, temp_beta = generate_eta_beta(n_pathway, sparsity, n_gene, case=case)
                eta[P] = temp_eta[P]
                beta[P] = temp_beta[P]
                useful_genes[P] = np.argwhere(eta[P] != 0).reshape(-1)
            
    return alpha, eta, beta, proportion, n_gene, n_pathway, n_class, useful_paths, useful_genes

