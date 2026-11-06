"""Report sections on the problem, the model and the architecture."""

from __future__ import annotations

from pathlib import Path

from eurosat_mlops import CLASSES
from eurosat_mlops.report.charts import FR_CLASSES
from eurosat_mlops.report.engine import Report
from eurosat_mlops.report.results import Results, fr, pct, thousands


def context(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    m = r.metrics
    pdf.h1("Contexte et choix du modèle")
    pdf.para(
        "Le sujet demande de mettre en production un modèle de machine learning qui ne soit pas "
        "trivial, avec quatre niveaux d'exigence, de 0 à 3. Nous avons retenu la classification "
        "d'occupation des sols sur le jeu **EuroSAT**. Ce sont des tuiles Sentinel-2 de 64 par 64 "
        "pixels, réparties en 10 classes (culture annuelle, culture permanente, pâturage, "
        "végétation herbacée, forêt, autoroute, zone industrielle, zone résidentielle, rivière, "
        "mer ou lac)."
    )
    pdf.para(
        "Ce n'est ni MNIST ni CIFAR-10, que le sujet écarte, et le problème n'est pas réglé "
        "d'avance. Vu du ciel, une culture annuelle, une culture permanente et un pâturage se "
        "ressemblent, et c'est exactement là que le modèle se trompe le plus, comme le montre la "
        "matrice de confusion plus bas. Une image satellite se prête aussi à une démonstration "
        "de dérive de données, puisqu'un voile de brume ou une surexposition se simule en une ligne."
    )
    pdf.h2("Le modèle et son entraînement")
    pdf.para(
        "Le modèle est un **ResNet18** pré-entraîné sur ImageNet, dont la dernière couche est "
        "remplacée par une couche à 10 sorties. Les réglages par défaut du script sont AdamW "
        "(taux d'apprentissage `1e-3`, décroissance de poids `1e-4`), un ordonnanceur OneCycle, "
        "un label smoothing de 0,05 et des augmentations simples, retournements et rotations d'un "
        "quart de tour, puisqu'une tuile satellite n'a pas d'orientation privilégiée. Le "
        "découpage est stratifié, 70 % pour l'entraînement, 15 % pour la validation et 15 % "
        "pour le test, avec une graine fixe (`42`) pour que chaque machine retrouve les mêmes "
        "ensembles."
    )
    history = m["history"]
    first, last = history[0], history[-1]
    pdf.para(
        f"L'historique enregistré pour le modèle livré compte {len(history)} époques, exécutées "
        f"sur {m['device'].upper()}. La précision de validation passe de "
        f"{pct(first['val_accuracy'])} à {pct(last['val_accuracy'])}. La première époque a duré "
        f"{thousands(round(first['seconds']))} s, les suivantes {thousands(round(history[1]['seconds']))} s "
        f"et {thousands(round(last['seconds']))} s. La normalisation ImageNet est intégrée au graphe du "
        f"modèle, qui prend des flottants entre 0 et 1. L'API, la calibration de la quantization "
        f"et l'entraînement partagent ainsi le même prétraitement, sans le recopier. L'export "
        f"ONNX s'écarte de PyTorch d'au plus {m['onnx_max_logit_gap']:.1e} sur les logits."
    )
    pdf.h2("Résultats sur le jeu de test")
    top = r.top_confusions(3)
    pdf.para(
        f"Sur les {thousands(m['samples'])} images de test, le modèle en virgule flottante atteint une "
        f"accuracy de **{pct(m['accuracy'])}** et un F1 macro de **{fr(m['macro_f1'], 4)}**. Les "
        f"classes les plus faciles sont la forêt et la mer ou le lac. Les plus difficiles sont "
        f"la culture permanente et la culture annuelle. La confusion la plus fréquente est une "
        f"culture permanente prise pour une culture annuelle ({top[0][2]} images), suivie "
        f"de l'inverse ({top[1][2]} images) et de culture annuelle prise pour pâturage "
        f"({top[2][2]} images)."
    )
    cm = r.confusion
    rows = [["Classe", "Images de test", "F1", "Erreurs"]]
    for i, name in enumerate(CLASSES):
        support = int(cm[i].sum())
        rows.append(
            [
                FR_CLASSES[i],
                str(support),
                fr(m["f1_per_class"][name], 3),
                str(support - int(cm[i, i])),
            ]
        )
    pdf.data_table(
        rows,
        [58, 36, 30, 30],
        "F1 par classe et nombre d'erreurs du modèle fp32 sur le jeu de test.",
        align="LRRR",
    )
    pdf.figure(
        figs["confusion"],
        "Matrice de confusion du modèle fp32 sur le jeu de test. La teinte indique la part de la "
        "classe réelle, le nombre est un effectif d'images.",
        width=118,
    )


def architecture(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Architecture")
    pdf.figure(
        figs["architecture"],
        "Les services de la stack. Le trait pointillé indique ce qui n'existe qu'en production.",
    )
    pdf.para(
        "Le navigateur et l'outil de test de charge passent par une **gateway nginx**, qui sert "
        "aussi l'interface statique et répartit les requêtes entre les répliques de l'API "
        "FastAPI. En production, Caddy se place devant pour le TLS. Les répliques écrivent "
        "l'usage dans **Postgres** et exposent leurs métriques à Prometheus. La route de dérive "
        "compare le trafic récent aux statistiques d'entraînement."
    )
    pdf.para(
        "Le modèle n'est pas copié dans l'image de l'API. Au démarrage, un service **registrar** "
        "publie le modèle ONNX quantifié dans **MLflow** et pose l'alias `production` s'il "
        "n'existe pas encore. Chaque réplique demande ensuite à MLflow la version désignée par "
        "cet alias et la charge à son démarrage. Promouvoir un modèle revient donc à déplacer "
        "l'alias, puis à redémarrer les répliques, car aucune ne recharge son modèle à chaud. "
        "Hors compose, la variable `MODEL_SOURCE=file` charge directement le fichier du dépôt."
    )
