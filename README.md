Ce projet s'inscrit dans le cadre du **PIRII (Projet d'Initiation à la Recherche et à l'Innovation)**. Il fournit une chaîne de traitement automatisée pour analyser les logs de sortie de récepteurs GNSS soumis à des campagnes de *Fuzz Testing*.

L'outil détecte automatiquement les instabilités du signal (Sauts, Dérives, Pics) qui ne seraient pas détectées par des méthodes statistiques classiques supposant un bruit purement Gaussien.

Le script analyse les séries temporelles (CSV) et identifie trois signatures de pannes :

1.  **Steps (Échelons) :** Détection par analyse du Gradient lissé et fusion temporelle des micro-sauts.
2.  **Ramps (Rampes) :** Détection de dérives lentes par segmentation et régression linéaire robuste.
3.  **Diracs (Pics) :** Détection d'outliers via le **Filtre de Hampel** (Médiane/MAD), insensible à l'effet de masquage.

## Prérequis

* **Python 3.8+**
* Les bibliothèques scientifiques suivantes :
Pandas
Numpy
Matplotlib
