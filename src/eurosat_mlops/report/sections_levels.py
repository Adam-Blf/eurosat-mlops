"""Report sections for levels 0, 1 and 2 of the subject."""

from __future__ import annotations

from pathlib import Path

from eurosat_mlops.report.engine import Report
from eurosat_mlops.report.results import DRIFT_FEATURES, Results, fr, thousands, users_per_run

FEATURE_FR = {
    "brightness": "luminosité",
    "mean_r": "moyenne rouge",
    "mean_g": "moyenne verte",
    "mean_b": "moyenne bleue",
    "contrast": "contraste",
}


def _sci(value: float) -> str:
    """Scientific notation with a decimal comma, for p-values."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa.replace('.', ',')}e{int(exponent)}"


def level0(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Niveau 0 : toute la stack en une commande")
    pdf.para(
        "Une seule commande démarre l'ensemble, base de données comprise : "
        "`docker compose up --build`. Chaque brique tourne dans son conteneur. Les ports ne sont "
        "ouverts que sur `127.0.0.1`, et le fichier `.env` (copié depuis `.env.example`) porte "
        "les jetons et le mot de passe de la base."
    )
    pdf.data_table(
        [
            ["Service", "Rôle", "Accès local"],
            ["postgres", "usage par utilisateur, base de MLflow", "interne"],
            ["mlflow", "registre de modèles, versions et alias", "127.0.0.1:5000"],
            ["registrar", "publie le modèle et pose l'alias production", "tâche unique"],
            ["api", "FastAPI, une à plusieurs répliques", "interne"],
            ["gateway", "nginx, interface et répartition", "127.0.0.1:8080"],
            ["prometheus", "métriques des répliques", "127.0.0.1:9090"],
            ["loadtest", "Locust, profil à activer", "à la demande"],
        ],
        [30, 100, 46],
        "Services de la stack compose.",
    )
    pdf.para(
        "L'intégration continue rejoue ce démarrage à chaque pull request. Elle construit toutes "
        "les images, lance la stack avec `docker compose up -d --wait`, puis vérifie trois choses "
        "au travers de la gateway : une requête sans jeton reçoit un 401, une image de forêt est "
        "bien classée `Forest` avec le jeton de démonstration, et la route d'usage connaît "
        "l'utilisateur de test. Les identifiants de démonstration sont des valeurs publiques. "
        "Le fichier de composition de production refuse de démarrer tant qu'elles n'ont pas été "
        "remplacées."
    )


def level1(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Niveau 1 : jeton, déploiement et test de charge")
    pdf.h2("Accès par jeton et suivi d'usage")
    pdf.para(
        "Le jeton se place dans l'en-tête `Authorization: Bearer <jeton>` (l'en-tête "
        "`X-API-Token` est accepté en repli). Sans jeton, l'API répond 401. Avec un jeton "
        "inconnu, elle répond 403. La comparaison se fait en temps constant sur tous les jetons "
        "enregistrés, pour qu'un temps de réponse ne révèle rien. Les jetons viennent de la "
        "variable `API_TOKENS` sous la forme `utilisateur:jeton`, et une liste vide refuse tout "
        "le monde. Un jeton administrateur distinct ouvre `/drift` et la vue complète de `/usage`."
    )
    pdf.para(
        "Chaque appel est écrit en base (date, utilisateur, route, code de réponse, classe, "
        "confiance, latence et cinq statistiques de l'image), ce qui sert à la fois au suivi "
        "d'usage et à la détection de dérive. Une limite de débit par jeton, "
        "1 200 requêtes par minute par défaut, protège l'API. Elle est tenue en mémoire par "
        "réplique, si bien que la limite réelle monte avec leur nombre. Les envois sont "
        "bornés à 2 Mo et aux formats JPEG, PNG, WEBP et TIFF."
    )
    pdf.data_table(
        [
            ["Route", "Accès", "Rôle"],
            ["GET /health, /ready", "libre", "sondes de vie et de disponibilité"],
            ["POST /predict", "jeton", "classe, probabilités, latence"],
            ["GET /model", "jeton", "version et alias du modèle chargé"],
            ["GET /usage", "jeton", "appels par utilisateur, tous pour l'admin"],
            ["GET /drift", "jeton admin", "verdict KS sur la fenêtre récente"],
            ["GET /metrics", "réseau interne", "métriques Prometheus"],
        ],
        [44, 32, 100],
        "Routes de l'API et niveau d'accès.",
    )
    pdf.h2("Déploiement automatique")
    pdf.para(
        "Le déploiement tient dans une commande, `python scripts/deploy.py --version 1.0.0`, "
        "pilotée par `DEPLOY_HOST` et `DEPLOY_ENV_FILE`. Le script refuse un fichier "
        "d'environnement incomplet ou qui contient encore des valeurs de démonstration. Il écrit "
        "le `.env` distant en mode 600 sans jamais l'afficher, lance la stack de production, puis "
        "attend la sonde de santé. L'option `--dry-run` affiche le plan sans contacter personne."
    )
    pdf.para(
        "Côté pipeline, le workflow `release.yml` se déclenche sur un tag `vX.Y.Z`. Il vérifie que "
        "le tag correspond à `pyproject.toml`, publie les trois images sur GHCR avec le numéro de "
        "version et le commit (jamais `latest`), puis crée la release avec les notes tirées du "
        "CHANGELOG. Le workflow `deploy.yml` prend le relais à la publication de la release, "
        "derrière un environnement `production` où l'on peut exiger une validation humaine."
    )
    pdf.para(
        "**Ce que nous n'avons pas fait.** Aucune VM de production réelle n'a été déployée. Le "
        "job de `deploy.yml` est ignoré tant que la variable `DEPLOY_ENABLED` n'est pas à `true`, "
        "ce qui garde un dépôt sans VM au vert. Le script est couvert par des tests unitaires et "
        "par son mode `--dry-run`, mais la séquence complète sur une machine distante n'a pas "
        "été jouée."
    )
    pdf.h2("Test de charge")
    users = users_per_run()
    total = sum(int(row["requests"]) for row in r.scaling)
    first, last = r.scaling[0], r.scaling[-1]
    gain = float(last["rps"]) / float(first["rps"]) - 1
    drop = 1 - float(last["p50"]) / float(first["p50"])
    pdf.para(
        "Le scénario Locust simule un analyste qui envoie l'une des dix tuiles d'exemple (une par "
        "classe) à `/predict`, en consultant de temps en temps sa propre consommation sur "
        "`/usage`, avec une pause de 0,05 à 0,3 s entre deux actions. Les jetons viennent de "
        "l'environnement et ne sont jamais affichés. Un garde-fou refuse toute cible qui n'est pas "
        "locale, afin de ne jamais tirer sur une production par erreur."
    )
    pdf.para(
        f"**Le test a tourné contre la stack compose locale, pas contre une production.** Nous "
        f"avons joué le même scénario avec {users} utilisateurs simulés et 1, 2 puis 4 répliques "
        f"de l'API, limiteur de débit désactivé pour la durée de la mesure. Sur les "
        f"{thousands(total)} requêtes des trois paliers, aucune n'a échoué."
    )
    rows = [["Répliques", "Requêtes", "Échecs", "req/s", "Médiane (ms)", "p95 (ms)", "p99 (ms)"]]
    for row in r.scaling:
        rows.append(
            [
                row["replicas"],
                thousands(int(row["requests"])),
                row["failures"],
                fr(float(row["rps"]), 1),
                row["p50"],
                row["p95"],
                row["p99"],
            ]
        )
    pdf.data_table(
        rows,
        [26, 26, 22, 22, 30, 25, 25],
        "Test de charge sur la route /predict selon le nombre de répliques de l'API.",
        align="CRRRRRR",
    )
    pdf.figure(figs["scaling"], "Débit et latences selon le nombre de répliques.", width=150)
    pdf.para(
        f"Passer de 1 à 4 répliques fait monter le débit de {fr(float(first['rps']), 1)} à "
        f"{fr(float(last['rps']), 1)} requêtes par seconde (+{fr(gain * 100, 0)} %) et baisse la "
        f"latence médiane de {first['p50']} à {last['p50']} ms (-{fr(drop * 100, 0)} %). Le gain "
        f"est modeste. Le générateur de charge, la gateway, la base et les répliques partagent "
        f"les mêmes coeurs de la même machine. Nous n'avons pas mesuré la répartition du CPU, c'est "
        f"l'explication que nous retenons. Un test sur une VM dédiée, avec le générateur sur une "
        f"autre machine, dirait davantage."
    )


def level2(pdf: Report, r: Results, figs: dict[str, Path]) -> None:
    pdf.h1("Niveau 2 : Terraform, interface, dérive et MLflow")
    pdf.h2("Terraform")
    pdf.para(
        "Le dossier `infra/terraform` décrit une VM Hetzner Cloud sous Ubuntu 24.04 où cloud-init "
        "installe Docker et le plugin compose, un pare-feu côté fournisseur et la clé SSH de "
        "déploiement. Le pare-feu n'ouvre le SSH qu'aux adresses de `admin_cidrs`, et une "
        "validation refuse `0.0.0.0/0`. L'utilisateur `deploy` n'est pas root et n'accepte que la "
        "clé, comme le suppose `scripts/deploy.py`. `ignore_changes` sur `user_data` évite qu'une "
        "retouche de cloud-init détruise une VM en marche."
    )
    pdf.para(
        "**La configuration passe terraform fmt -check et terraform validate, mais elle n'a "
        "jamais été appliquée**, parce qu'un `apply` crée une ressource payante (environ 4 euros "
        "par mois pour un cx22). L'état est local et ignoré par git. Au-delà d'une démonstration "
        "d'école, il faudrait un backend distant."
    )
    pdf.h2("Interface web")
    pdf.para(
        "L'interface est une page statique servie par la gateway. On y glisse une tuile, on colle "
        "son jeton, et elle affiche la classe en français avec la confiance. Les erreurs 401, 403, "
        "413, 415 et 429 ont chacune un message lisible. Rien ne vient d'un CDN : les icônes "
        "et les polices (Epilogue et Overpass Mono, les mêmes que ce rapport) sont embarquées, "
        "et la politique de sécurité de contenu n'autorise que l'origine de la page."
    )
    pdf.h2("Détection de dérive")
    pdf.para(
        "La référence est un sous-échantillon de 2 000 images d'entraînement, résumées par cinq "
        "statistiques : luminosité, moyenne de chaque canal rouge, vert et bleu, et contraste. La "
        "route `/drift` compare la fenêtre des images récentes (200 par défaut) à cette référence "
        "avec un test de Kolmogorov-Smirnov à deux échantillons par statistique. Le seuil de "
        "0,01 est divisé par le nombre de statistiques, 5, d'après la correction de Bonferroni, "
        "soit 0,002. Sous 30 images, le détecteur ne rend aucun verdict. Le résultat alimente "
        "aussi des jauges Prometheus."
    )
    clean, bad = r.drift["clean"], r.drift["degraded"]
    pdf.para(
        f"La démonstration tire {clean['samples']} images du jeu de test, une première fois "
        f"intactes, une seconde fois dégradées par un voile de brume et une surexposition "
        f"(chaque pixel est multiplié par 0,6 puis augmenté de 95). Sur le trafic intact, aucune "
        f"statistique n'est signalée, la plus basse p-value étant "
        f"{fr(min(f['p_value'] for f in clean['features'].values()), 2)}. Sur le trafic dégradé, "
        f"toutes les statistiques ({len(bad['drifted_features'])} sur 5) sont signalées, avec des "
        f"p-values d'au plus {_sci(max(f['p_value'] for f in bad['features'].values()))}."
    )
    rows = [
        [
            "Statistique",
            "Réf.",
            "Intact",
            "KS",
            "p-value",
            "Dégradé",
            "KS",
            "p-value",
        ]
    ]
    for f in DRIFT_FEATURES:
        c, d = clean["features"][f], bad["features"][f]
        rows.append(
            [
                FEATURE_FR[f],
                fr(c["reference_mean"], 3),
                fr(c["current_mean"], 3),
                fr(c["ks_statistic"], 3),
                fr(c["p_value"], 3),
                fr(d["current_mean"], 3),
                fr(d["ks_statistic"], 3),
                _sci(d["p_value"]),
            ]
        )
    pdf.data_table(
        rows,
        [34, 17, 19, 17, 21, 19, 17, 32],
        "Test KS sur 300 images, trafic intact puis dégradé. Réf. désigne la moyenne de la "
        "référence d'entraînement, toutes les moyennes sont comprises entre 0 et 1.",
        align="LRRRRRRR",
    )
    pdf.figure(
        figs["drift"], "Statistique KS par caractéristique, trafic intact et dégradé.", width=140
    )
    pdf.para(
        "Cette démonstration est volontairement grossière. Elle montre que le détecteur reste "
        "silencieux sur un trafic normal et réagit à un décalage massif, pas qu'il repère des "
        "dérives fines. Il ne regarde que des statistiques de couleur. Un changement de saison "
        "ou de région qui garderait la même luminosité moyenne passerait inaperçu."
    )
    pdf.h2("MLflow : versions et alias")
    pdf.para(
        "Le module `registry.py` s'appuie sur les alias de MLflow. La commande `publish` journalise "
        "le modèle ONNX, les statistiques de référence et les métriques dans un run, enregistre une "
        "nouvelle version de `eurosat-classifier` et lui pose l'alias `candidate`. La commande "
        "`promote` déplace l'alias `production` vers cette version, à condition que son accuracy "
        "de test ne soit pas inférieure de plus de 0,005 à celle de la version en place, sauf "
        "avec `--force`. Le service `registrar` de la stack appelle `bootstrap`, qui publie puis "
        "promeut seulement si aucun alias `production` n'existe, ce qui rend le démarrage "
        "répétable sans empiler de versions."
    )
