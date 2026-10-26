# Courses de billes pour TikTok

Génère des vidéos verticales (1080x1920, 30 fps, 28 à 48 s) avec son synthétisé, légende et hashtags.
Les textes de la vidéo et les légendes sont **en anglais par défaut** (`--lang fr` pour le français).
Tu publies toi-même : le script prépare les brouillons.

## Installation (une seule fois)

```bash
# 1. ffmpeg
sudo apt install ffmpeg        # Linux
brew install ffmpeg            # macOS
# Windows : winget install ffmpeg

# 2. Python 3.10+
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python generate.py --count 1 --fast        # test rapide
python generate.py                         # 7 vidéos à partir de demain
python generate.py --start 2026-11-02 --count 7 --series "#MaChaine"
python generate.py --lang fr               # textes et légendes en français
python generate.py --min-dur 20 --max-dur 35   # courses plus courtes
```

Le rendu prend environ 1 à 2 minutes par vidéo (`--fast` pour un test plus rapide, de moindre qualité).

Résultat dans `out/` :

```
out/2026-10-27/video.mp4     vidéo à uploader
out/2026-10-27/caption.txt   légende + hashtags + commentaire épinglé suggéré
out/2026-10-27/meta.json     seed, thème, gagnant, durée
out/planning.csv             récapitulatif du lot
```

- Une même date donne toujours la même vidéo (seed = date), donc tu peux la régénérer à l'identique.
- Les vidéos déjà générées sont ignorées (`--force` pour refaire).
- Un thème visuel différent chaque jour de la semaine.
- Chaque course enchaîne 12 à 15 sections tirées au hasard : rampes en zigzag, champ de plots (plinko),
  entonnoirs, hélices rotatives. Deux vidéos n'ont donc jamais le même parcours.
- Le script retente automatiquement d'autres variantes d'un parcours tant que la durée n'est pas dans la
  plage visée.

## Routine hebdomadaire (15 min)

1. Dimanche : `python generate.py --count 7`
2. Envoie les vidéos sur ton téléphone (AirDrop, Google Drive, Telegram « messages enregistrés »).
3. Dans TikTok : importe la vidéo, ajoute un son tendance à faible volume (ça booste la portée et ne peut
   se faire que dans l'app), colle la légende de `caption.txt`, choisis une miniature, publie ou enregistre
   en brouillon.
4. Option : TikTok Studio sur ordinateur permet de programmer des publications à l'avance (vérifie la
   limite actuelle). Tu peux ainsi planifier toute la semaine en une fois.
5. Mets le « commentaire épinglé suggéré » dans les commentaires après publication.
6. Publie aussi la même vidéo en YouTube Shorts et Instagram Reels.

## Génération automatique chaque semaine (optionnel)

Le workflow `.github/workflows/weekly.yml` génère les 7 vidéos tous les dimanches sur GitHub Actions
et les met à disposition en téléchargement (onglet Actions > run > Artifacts, conservé 14 jours).
Pousse le projet sur un dépôt GitHub, c'est actif automatiquement. Tu peux aussi le lancer à la main
avec « Run workflow ».

## Personnaliser

Tout est dans `generate.py`, section « Paramètres généraux » :

- `THEMES` : couleurs des décors
- `BALL_NAMES` / `BALL_COLORS` : noms et couleurs des billes
- `CAPTIONS` / `HASHTAGS` : légendes et hashtags
- `MIN_DUR` / `MAX_DUR` : durée visée (aussi via `--min-dur` / `--max-dur`)
- `SECTIONS` : nombre de sections du circuit, plus il est grand plus la course est longue
- `GRAVITY` : vitesse générale des billes
- `build_track()` : les types de sections (ajoute-en une en écrivant une fonction et en la
  mettant dans la liste `kinds`)

## Pistes d'amélioration

- Obstacles mobiles (pales rotatives, plateformes) avec `pymunk.Body.KINEMATIC`
- Noms de billes = pseudos de tes abonnés (engagement)
- Publication automatique via la Content Posting API de TikTok (nécessite un audit de l'app, les
  vidéos publiées avant audit restent privées)
