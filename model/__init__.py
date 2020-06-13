from .base import ModelSML, ModelNN, ModelGNN
from .sml import ModelLogisticRegression, ModelPerceptron, ModelRidgeClassifier, ModelLasso, ModelSGDClassifier, \
    ModelBernoulliNB, ModelMultinomialNB, ModelGaussianNB, ModelDecisionTreeClassifier, \
    ModelGradientBoostingClassifier, ModelRandomForestClassifier, ModelXGBClassifier, \
    ModelCatBoostClassifier, ModelLGBMClassifier
from .deepfm import DeepFM
from .stg import STG
from .vime import VIME
from .tabnet import TabNet


MODEL_DICT = {
    "ModelSML": ModelSML, "SML": ModelSML,
    "ModelNN": ModelNN, "NN": ModelNN,
    "ModelGNN": ModelGNN, "GNN": ModelGNN,
    "ModelLogisticRegression": ModelLogisticRegression, "LogisticRegression": ModelLogisticRegression,
    "ModelPerceptron": ModelPerceptron, "Perceptron": ModelPerceptron,
    "ModelRidgeClassifier": ModelRidgeClassifier, "RidgeClassifier": ModelRidgeClassifier,
    "ModelLasso": ModelLasso, "Lasso": ModelLasso,
    "ModelSGDClassifier": ModelSGDClassifier, "SGDClass