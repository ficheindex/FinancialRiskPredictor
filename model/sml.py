
# -*- coding: utf-8 -*-

import numpy as np
from model.base import ModelSML


class ModelLogisticRegression(ModelSML):

    def __init__(self, args,
                 random_state=None, C=None, solver=None, max_iter=None):
        super().__init__(args)

        from sklearn.linear_model import LogisticRegression

        self.random_state = random_state
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            "max_iter": [100, 1000],
        }

        self.model = LogisticRegression(
            penalty="l2",
            dual=False,
            tol=1e-4,
            C=1.0 if self.C is None else self.C,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            random_state=args.seed if self.random_state is None else self.random_state,
            solver="lbfgs" if self.solver is None else self.solver,
            max_iter=1000 if self.max_iter is None else self.max_iter,  # original: 100
            multi_class="auto",
            verbose=0,
            warm_start=False,
            n_jobs=-1,  # default: None
            l1_ratio=None,
        )


class ModelPerceptron(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import Perceptron

        self.model = Perceptron(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            n_jobs=-1,  # default: None
        )


class ModelRidgeClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import RidgeClassifier

        self.model = RidgeClassifier(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
        )


class ModelLasso(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import Lasso

        self.model = Lasso(
            random_state=args.seed,
        )
        # class_weight=args.class_weight if hasattr(args, "class_weight") else None,


class ModelSGDClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.linear_model import SGDClassifier

        self.model = SGDClassifier(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            n_jobs=-1,  # default: None
        )


class ModelBernoulliNB(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.naive_bayes import BernoulliNB

        self.model = BernoulliNB()


class ModelMultinomialNB(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.naive_bayes import MultinomialNB

        self.model = MultinomialNB()


class ModelGaussianNB(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.naive_bayes import GaussianNB

        self.model = GaussianNB()


class ModelDecisionTreeClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.tree import DecisionTreeClassifier

        self.model = DecisionTreeClassifier(
            random_state=args.seed,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
        )


class ModelGradientBoostingClassifier(ModelSML):

    def __init__(self, args):
        super().__init__(args)

        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            random_state=args.seed,
        )
        # class_weight=args.class_weight if hasattr(args, "class_weight") else None,


class ModelRandomForestClassifier(ModelSML):

    def __init__(self, args,
                 random_state=None, n_estimators=None, max_depth=None):
        super().__init__(args)

        from sklearn.ensemble import RandomForestClassifier

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "n_estimators": list(range(80, 200, 10)),
            "max_depth": list(range(2, 15, 1)),
        }

        self.model = RandomForestClassifier(
            n_estimators=100 if self.n_estimators is None else self.n_estimators,
            criterion="gini",
            max_depth=None if self.max_depth is None else self.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,  # default: None
            random_state=args.seed if self.random_state is None else self.random_state,
            verbose=0,
            warm_start=False,
            class_weight=args.class_weight if hasattr(args, "class_weight") else None,
            ccp_alpha=0.0,
            max_samples=None,
        )


class ModelXGBClassifier(ModelSML):

    def __init__(self, args,
                 random_state=None, n_estimators=None, max_depth=None, learning_rate=None,
                 subsample=None, colsample_bytree=None, min_child_weight=None):
        super().__init__(args)

        from xgboost import XGBClassifier

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight

        self.param_grid = {
            "random_state": [0, 1, 42, 1234],
            "n_estimators": list(range(80, 200, 10)),
            "max_depth": list(range(2, 15, 1)),
            "learning_rate": list(np.linspace(0.01, 2, 20)),
            "subsample": list(np.linspace(0.7, 1.0, 20)),
            "colsample_bytree": list(np.linspace(0.5, 1.0, 10)),
            "min_child_weight": list(range(1, 20, 2)),
        }

        self.model = XGBClassifier(
            max_depth=None if self.max_depth is None else self.max_depth,
            max_leaves=None,
            max_bin=None,
            grow_policy=None,
            learning_rate=None if self.learning_rate is None else self.learning_rate,
            n_estimators=100 if self.n_estimators is None else self.n_estimators,
            verbosity=None,
            objective=None,
            booster=None,
            tree_method=None,
            n_jobs=-1,  # default: None
            gamma=None,
            min_child_weight=None if self.min_child_weight is None else self.min_child_weight,
            max_delta_step=None,
            subsample=None if self.subsample is None else self.subsample,
            sampling_method=None,
            colsample_bytree=None if self.colsample_bytree is None else self.colsample_bytree,
            colsample_bylevel=None,
            colsample_bynode=None,
            reg_alpha=None,
            reg_lambda=None,