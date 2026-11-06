"""Report sections for level 3, security, limits and conclusion."""

from __future__ import annotations

from pathlib import Path

from eurosat_mlops.report.engine import Report
from eurosat_mlops.report.results import Results, fr, thousands

MODEL_LABELS = {"fp32": "fp32", "int8 dynamic": "int8 dynamique", "int8 static": "int8 statique"}


def level3(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Niveau 3 : quantization, scaling et canary")
    pdf.h2("Quantization")
    q, n = r.quant, r.metrics["samples"]
    pdf.para(
        "Nous avons comparé trois variantes ONNX du même modèle. La quantization **statique** "
        "(format QDQ, poids int8 par canal, activations uint8) est calibrée sur 256 images "
        "d'entraînement. La quantization **dynamique** ne convertit que les poids en int8 et "
        f"calcule les échelles des activations à chaque appel. Les trois sont évaluées sur les "
        f"{thousands(n)} images de test. La latence est mesurée en batch 1 sur un seul fil, médiane et "
        f"p95 sur 200 appels après 20 appels d'échauffement."
    )
    rows = [["Modèle", "Taille (Mo)", "Accuracy", "F1 macro", "Médiane (ms)", "p95 (ms)", "Accél."]]
    base = q["fp32"]["median_ms"]
    for key, v in q.items():
        rows.append(
            [
                MODEL_LABELS[key],
                fr(v["size_mb"], 2),
                fr(v["accuracy"], 4),
                fr(v["macro_f1"], 4),
                fr(v["median_ms"], 2),
                fr(v["p95_ms"], 2),
                fr(base / v["median_ms"], 2) + "x",
            ]
        )
    pdf.data_table(
        rows,
        [32, 24, 24, 24, 28, 22, 22],
        "Les trois modèles sur le jeu de test. Le modèle servi est l'int8 statique (en gras).",
        align="LRRRRRR",
        bold_row=3,
    )
    pdf.figure(figs["quantization"], "Taille et latence médiane des trois modèles.", width=150)
    size_cut = q["fp32"]["size_mb"] / q["int8 static"]["size_mb"]
    slow = q["int8 dynamic"]["median_ms"] / base
    lost = r.correct("fp32") - r.correct("int8 static")
    gained = r.correct("int8 dynamic") - r.correct("fp32")
    pdf.para(
        f"Les deux variantes divisent la taille du fichier par {fr(size_cut, 1)}. Côté qualité, "
        f"la version statique classe correctement {lost} images de moins que le fp32 sur {thousands(n)}, la "
        f"dynamique en classe {gained} de plus. Ces écarts sont de l'ordre de quelques images et "
        f"nous ne les lisons ni comme une perte ni comme un gain réels."
    )
    pdf.para(
        f"Côté vitesse, les deux variantes ne se comportent pas du tout pareil. La version "
        f"statique est {fr(base / q['int8 static']['median_ms'], 2)} fois plus rapide que le fp32, "
        f"et réduit aussi l'écart entre la médiane et le p95. La version dynamique est "
        f"{fr(slow, 0)} fois **plus lente** que le fp32 (médiane de {fr(q['int8 dynamic']['median_ms'], 0)} ms). "
        f"Nous n'avons pas profilé son graphe. L'explication la plus plausible est que, sur CPU, "
        f"les convolutions d'un ResNet ne trouvent pas de noyau int8 efficace en quantization "
        f"dynamique, alors qu'il faut en plus calculer les échelles des activations à chaque "
        f"appel. En statique, les échelles sont figées à la calibration et le runtime enchaîne "
        f"des opérations int8. On retient qu'une taille réduite ne garantit pas un modèle plus rapide, et que "
        f"cela se mesure sur la cible. C'est le modèle statique qui est servi."
    )
    pdf.h2("Scaling horizontal")
    pdf.para(
        "Le nombre de répliques de l'API se règle avec `API_REPLICAS` (2 par défaut). La gateway "
        "nginx s'appuie sur le DNS de Docker, qui renvoie toutes les répliques, avec un délai de "
        "résolution de 5 secondes pour suivre celles qui apparaissent ou disparaissent. Prometheus "
        "interroge chaque réplique directement, et l'usage va dans une base Postgres commune, de "
        "sorte que les décomptes par utilisateur restent cohérents d'une réplique à l'autre. "
        "Les mesures du niveau 1 (1, 2 et 4 répliques) sont celles de ce scaling, avec la réserve "
        "déjà dite sur la machine partagée."
    )
    pdf.h2("Canary sur Kubernetes")
    pdf.para(
        "Le dossier `k8s` contient une base kustomize (namespace, Postgres, MLflow, job du "
        "registre, API, gateway, ingress) et deux overlays. L'overlay `local` déploie la stack "
        "complète. L'overlay `canary` ajoute un déploiement `api-canary` d'un seul pod qui charge "
        "l'alias `candidate` au lieu de `production`, avec son propre service et un second "
        "ingress."
    )
    pdf.para(
        "Le partage du trafic repose sur les annotations de canary d'ingress-nginx. Un poids de "
        "10 envoie environ 10 % des requêtes vers le candidat, et tout appel portant l'en-tête "
        "`X-Canary: always` le joint à coup sûr, ce qui permet de tester un candidat à la main. "
        "Quand il est validé, on déplace l'alias `production` dans MLflow, on redémarre les "
        "répliques et on retire l'overlay. Les pods tournent en non-root, système de fichiers "
        "en lecture seule, toutes les capacités retirées, avec des sondes `/ready` et `/health`."
    )
    pdf.para(
        "**Ces manifestes n'ont pas été éprouvés sur un cluster.** Aucun résultat du dépôt ne "
        "documente une exécution sous Kubernetes, et la bascule se fait à la main, sans analyse "
        "automatique des métriques du candidat."
    )


def quality(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Qualité, observabilité et configuration")
    pdf.para(
        "L'intégration continue tourne à chaque pull request et sur `main`. Un premier job "
        "exécute `ruff check` et `ruff format --check`. Un deuxième lance `pytest`, dont les "
        "tests d'authentification, d'usage, de dérive, de déploiement et du garde-fou de test de "
        "charge. Un troisième construit toutes les images et rejoue le démarrage et le test de "
        "fumée décrits au niveau 0. Le script `sync_version.py` vérifie que le badge du README, "
        "le CHANGELOG et `pyproject.toml` annoncent la même version."
    )
    pdf.para(
        "Chaque réplique expose cinq séries Prometheus : les requêtes par utilisateur, route et "
        "code, les prédictions par classe, un histogramme du temps d'inférence (de 2 ms à 1 s), "
        "la p-value de dérive par statistique et un indicateur de dérive détectée. Prometheus "
        "découvre les répliques par le DNS de Docker. La route `/metrics` n'est pas joignable par "
        "la gateway publique."
    )
    pdf.data_table(
        [
            ["Variable", "Rôle", "Défaut"],
            [
                "API_TOKENS",
                "jetons clients, utilisateur:jeton séparés par des virgules",
                "vide, tout refusé",
            ],
            ["ADMIN_TOKEN", "jeton admin pour /drift et /usage global", "vide"],
            ["API_REPLICAS", "nombre de répliques de l'API", "2"],
            ["MODEL_SOURCE", "file ou mlflow", "file (mlflow en compose)"],
            ["MODEL_ALIAS", "alias MLflow chargé au démarrage", "production"],
            ["RATE_LIMIT_PER_MINUTE", "limite par jeton et par réplique", "1200"],
            ["MAX_UPLOAD_BYTES", "taille maximale d'une image", "2 000 000"],
            ["DRIFT_WINDOW", "images récentes comparées à la référence", "200"],
        ],
        [48, 88, 40],
        "Principales variables d'environnement.",
        size=7.8,
    )


def security(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Choix de sécurité")
    pdf.para(
        "Nous avons traité la sécurité à chaque couche plutôt qu'en fin de projet. Le tableau "
        "résume les choix et l'endroit où ils vivent."
    )
    pdf.data_table(
        [
            ["Sujet", "Choix", "Où"],
            [
                "Accès",
                "jeton obligatoire, 401 sans jeton, 403 si inconnu, comparaison en temps constant",
                "api/auth.py",
            ],
            [
                "Fichiers envoyés",
                "formats en liste blanche (SVG exclu), 2 Mo, 25 millions de pixels au plus",
                "preprocess.py, nginx",
            ],
            [
                "Secrets",
                "dans des fichiers ignorés par git et dans les secrets GitHub, jamais dans le "
                "dépôt. Le déploiement refuse les valeurs de démonstration",
                "scripts/deploy.py",
            ],
            [
                "Conteneurs",
                "système de fichiers en lecture seule, capacités retirées, non-root, "
                "ports sur 127.0.0.1",
                "docker-compose.yml",
            ],
            [
                "Entrée web",
                "en-têtes nosniff, DENY, CSP limitée à l'origine, limite de débit, "
                "route de métriques masquée, CORS restreint, TLS par Caddy en production",
                "gateway, Caddyfile",
            ],
            [
                "Test de charge",
                "refus de toute cible non locale et du mode interface web",
                "loadtest/guard.py",
            ],
            [
                "Chaîne de livraison",
                "actions GitHub épinglées par empreinte, images versionnées, jamais latest",
                "workflows",
            ],
            [
                "Réseau cloud",
                "pare-feu fournisseur, SSH limité à admin_cidrs, connexion par clé",
                "infra/terraform",
            ],
        ],
        [30, 104, 42],
        "Mesures de sécurité du projet.",
        size=7.6,
    )


def closing(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Limites et suites possibles")
    pdf.para("Ce que le projet ne prouve pas, et ce que nous ferions ensuite.")
    pdf.bullets(
        [
            "**Pas de production réelle.** Terraform est validé mais jamais appliqué, et le "
            "workflow de déploiement est resté inactif. La chaîne VM, déploiement, test n'a pas "
            "été exécutée de bout en bout. La suite logique est d'appliquer Terraform pour une "
            "séance, de déployer, puis de détruire.",
            "**Charge mesurée en local.** Les chiffres du scaling viennent d'une seule machine "
            "partagée avec le générateur de charge. Ils comparent des configurations entre elles "
            "mais ne disent rien d'une production. Nous les refairions sur une VM dédiée.",
            f"**Entraînement court.** {len(r.metrics['history'])} époques sur CPU, un seul "
            f"découpage et une seule graine, sans intervalle de confiance. Nous n'avons pas "
            f"comparé d'autres architectures. Un entraînement sur GPU avec plusieurs graines "
            f"serait plus solide (le notebook Colab permet de s'entraîner sur GPU).",
            "**Dérive limitée aux couleurs.** Le détecteur ne regarde ni les prédictions ni "
            "les étiquettes. En vrai, il faudrait suivre la distribution des classes prédites et "
            "relier la dérive à un réentraînement.",
            "**Limiteur de débit par réplique.** Il vit en mémoire, il faudrait le sortir vers un "
            "stockage partagé. De même, stocker des empreintes de jetons en base serait mieux que "
            "la variable d'environnement.",
            "**Canary à la main.** Les manifestes ne sont pas testés sur un cluster et rien ne "
            "bascule ni ne revient en arrière automatiquement. On pourrait brancher la décision "
            "sur les métriques Prometheus.",
            "**Quantization dynamique.** Notre explication de sa lenteur reste une hypothèse. Un "
            "profilage avec ONNX Runtime la confirmerait ou la corrigerait.",
        ]
    )
    pdf.h1("Conclusion")
    pdf.para(
        f"Nous avons mis en production un classifieur d'images satellite qui tient dans une "
        f"seule commande, derrière une API à jeton avec suivi d'usage, un registre de modèles, "
        f"un détecteur de dérive et une interface. Le modèle servi pèse "
        f"{fr(r.quant['int8 static']['size_mb'], 1)} Mo pour une accuracy de "
        f"{fr(r.quant['int8 static']['accuracy'] * 100, 2)} % sur le jeu de test."
    )
    pdf.para(
        f"Le plus instructif a été de mesurer plutôt que de supposer. La quantization dynamique, "
        f"qui réduit la taille aussi bien que la statique, ralentit le modèle d'un facteur "
        f"{fr(r.quant['int8 dynamic']['median_ms'] / r.quant['fp32']['median_ms'], 0)}, et passer "
        f"à quatre répliques ne gagne qu'une partie de ce que l'on attendait. Les deux "
        f"résultats nous ont appris à garder une mesure derrière chaque choix, et à écrire "
        f"clairement ce qui n'a pas été fait."
    )
