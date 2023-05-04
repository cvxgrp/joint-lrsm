import numpy as np
import networkx as nx
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


def set_edge_weight(G, weight, random=False, std=1.0):
    if random:
        for _, _, e in G.edges(data=True):
            e["weight"] = max(np.random.normal(weight, std), 0)
    else:
        for _, _, e in G.edges(data=True):
            e["weight"] = weight


def cartesian_product(graphs):
    """Performs a cartesian product between a list of networkx graphs."""
    G = nx.cartesian_product(graphs[0], graphs[1])
    for i in range(2, len(graphs)):
        G = nx.cartesian_product(G, graphs[i])
    mapping = {}
    for node in G.nodes():
        mapping[node] = tuple(flatten(node))
    return nx.relabel_nodes(G, mapping)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def get_data_graph(dataset):
    pd.options.mode.chained_assignment = None

    if dataset == 'mesothelioma':
        df = pd.read_excel(r"data\Mesothelioma data set.xlsx")
        df['age'] = pd.to_numeric(df['age']).astype(np.int64)
        df = pd.get_dummies(
            df,
            columns=['city', 'keep side', 'type of MM', 'habit of cigarette'])

        df_train, df_test = model_selection.train_test_split(df, test_size=0.1)

        list_of_ages = np.arange(19, 86)
        G_sex = nx.Graph()
        G_sex.add_nodes_from(['Male', 'Female'])
        G_sex.add_edge('Male', 'Female')
        num_ages = len(list_of_ages)
        G_age = nx.path_graph(num_ages)
        index_to_age = dict(zip(np.arange(num_ages), list_of_ages))
        G_age = nx.relabel_nodes(G_age, index_to_age)
        set_edge_weight(G_sex, 10)
        set_edge_weight(G_age, 500)
        G = cartesian_product([G_sex, G_age])

        X_train, Y_train, Z_train, Cls_train = mesothelioma_data(df_train, G)
        X_test, Y_test, Z_test, Cls_test = mesothelioma_data(df_test, G)

    elif dataset == 'wine':
        df = pd.read_csv(r"data\wine.csv")

        bins = 10
        df['density_bin'] = pd.cut(df['density'], bins=bins)
        df['density_bin'] = df['density_bin'].cat.codes
        df['sulphates_bin'] = pd.cut(df['sulphates'], bins=bins)
        df['sulphates_bin'] = df['sulphates_bin'].cat.codes

        df_train, df_test = model_selection.train_test_split(df, test_size=0.2)

        G_density = nx.path_graph(bins)
        G_sulphates = nx.path_graph(bins)
        # set_edge_weight(G_density, 20, random=False)
        # set_edge_weight(G_sulphates, 10, random=False)
        set_edge_weight(G_density, 0, random=False)
        set_edge_weight(G_sulphates, 0, random=False)
        G = cartesian_product([G_density, G_sulphates])

        X_train, Y_train, Z_train, Cls_train = wine_data(df_train, G)
        X_test, Y_test, Z_test, Cls_test = wine_data(df_test, G)

    elif dataset == 'election':
        raw_data = pd.read_csv(r"data\1976-2016-senate.csv")
        df = raw_data.copy()
        df['democrat'] = (df['party'] == 'democrat').astype(np.double)
        df.drop([
            'candidate', 'writein', 'state', 'state_ic', 'state_fips',
            'state_cen', 'office', 'version', 'stage', 'special', 'party',
            'totalvotes'
        ],
                inplace=True,
                axis=1)
        states = df.state_po.unique()
        years = np.sort(df.year.unique())
        neighbors = pd.read_csv(r"data\neighbors-states.csv")

        df_train = df.query('year != 2014 & year != 2016')
        df_test = df.query('year == 2014 | year == 2016')

        def election_data(df):
            Y = []
            Z = []
            Cls = []
            for i, (state, year) in enumerate([(state, year)
                                               for state in states
                                               for year in years]):
                data = df.loc[(raw_data.year == year) & (df.state_po == state)]
                for district in data.district.unique():
                    data_dist = data.loc[df.district == district]
                    Y.append(
                        data_dist.democrat[data_dist.candidatevotes.idxmax()])
                    Z.append((state, year))
                    Cls.append(i)
            return Y, Z, Cls

        Y_train, Z_train, Cls_train = election_data(df_train)
        Y_test, Z_test, Cls_test = election_data(df_test)

        data_train = dict(Y=Y_train, Z=Z_train, Cls=Cls_train)
        data_test = dict(Y=Y_test, Z=Z_test, Cls=Cls_test)

        G_state = nx.Graph()
        for state in states:
            G_state.add_node(state)

        for state1 in states:
            for state2 in states:
                if state2 in list(neighbors[neighbors.StateCode == state1]
                                  ['NeighborStateCode']):
                    G_state.add_edge(state1, state2)

        n_years = len(years)
        G_time = nx.path_graph(n_years)
        G_time = nx.relabel_nodes(G_time, dict(zip(np.arange(n_years), years)))
        set_edge_weight(G_state, 1)
        set_edge_weight(G_time, 4)
        G = cartesian_product([G_state, G_time])

        return data_train, data_test, G

    elif dataset == 'concrete':
        df = pd.read_csv(r"data\Concrete_Data.csv")
        bins = 10
        df['age_bin'] = pd.cut(df['Age (day)'], bins=bins)
        df['ash_bin'] = pd.cut(
            df['Fly Ash (component 3)(kg in a m^3 mixture)'], bins=bins)
        df['age_bin'] = df['age_bin'].cat.codes
        df['ash_bin'] = df['ash_bin'].cat.codes

        df_train, df_test = model_selection.train_test_split(df,
                                                             test_size=0.25)

        G_age = nx.path_graph(bins)
        G_ash = nx.path_graph(bins)
        # set_edge_weight(G_age, .5, random=False)
        # set_edge_weight(G_ash, .5, random=False)
        set_edge_weight(G_age, 0, random=False)
        set_edge_weight(G_ash, 0, random=False)
        G = cartesian_product([G_age, G_ash])

        X_train, Y_train, Z_train, Cls_train = concrete_data(df_train, G)
        X_test, Y_test, Z_test, Cls_test = concrete_data(df_test, G)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    data_train = dict(X=X_train, Y=Y_train, Z=Z_train, Cls=Cls_train)
    data_test = dict(X=X_test, Y=Y_test, Z=Z_test, Cls=Cls_test)
    return data_train, data_test, G


def wine_data(df, G):
    Xs = []
    Ys = []
    Zs = []
    Cls = []
    for i, node in enumerate(G.nodes()):
        densitybin, sulphatesbin = node
        df_node = df.query('density_bin == %d & sulphates_bin == %d' %
                           (densitybin, sulphatesbin))
        X_node = np.array(df_node.drop([
            'sulphates', 'density', 'density_bin', 'sulphates_bin', 'quality',
            'alcohol'
        ],
                                       axis=1),
                          dtype=float)
        Y_node = df_node['quality'].values
        Y_node[Y_node == 'good'] = 1.
        Y_node[Y_node == 'bad'] = -1.
        Y_node = Y_node.astype(float)

        N = X_node.shape[0]
        Xs += [X_node]
        Ys += [Y_node]
        Zs.extend([node] * N)
        Cls.extend([i] * N)

    return np.concatenate(Xs,
                          axis=0), np.concatenate(Ys,
                                                  axis=0)[:,
                                                          np.newaxis], Zs, Cls


def mesothelioma_data(df, G):
    Xs = []
    Ys = []
    Zs = []
    Cls = []
    for i, node in enumerate(G.nodes()):
        sex, age = node
        gender = 1 if sex == 'Male' else 0
        df_node = df.query('age == %d & gender == %d' % (age, gender))
        X_node = np.array(df_node.drop(
            ['age', 'gender', 'class of diagnosis', 'diagnosis method'],
            axis=1),
                          dtype=float)
        Y_node = np.array(df_node['class of diagnosis'], dtype=float)

        N = X_node.shape[0]
        Xs += [X_node]
        Ys += [Y_node]
        Zs.extend([node] * N)
        Cls.extend([i] * N)

    return np.concatenate(Xs,
                          axis=0), np.concatenate(Ys,
                                                  axis=0)[:,
                                                          np.newaxis], Zs, Cls


def concrete_data(df, G):
    Xs = []
    Ys = []
    Zs = []
    Cls = []
    for i, node in enumerate(G.nodes()):
        agebin, ashbin = node
        df_node = df.query('age_bin == %d & ash_bin == %d' % (agebin, ashbin))
        X_node = np.array(
            df_node.drop([
                'Age (day)', 'Fly Ash (component 3)(kg in a m^3 mixture)',
                'age_bin', 'ash_bin',
                'Concrete compressive strength(MPa, megapascals) '
            ],
                         axis=1))
        Y_node = np.array(
            df_node['Concrete compressive strength(MPa, megapascals) '])
        N = X_node.shape[0]
        Xs += [X_node]
        Ys += [Y_node]
        Zs.extend([node] * N)
        Cls.extend([i] * N)

    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), Zs, Cls
