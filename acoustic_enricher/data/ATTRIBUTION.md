# Bundled vocabulary data

| File | Source | Licence |
|---|---|---|
| `audioset.json` | Google AudioSet ontology, https://github.com/audioset/ontology (`ontology.json`, reduced to id/name/child_ids/restrictions) | CC BY-SA 4.0 — © Google LLC. Gemmeke, J. F. et al. (2017), "Audio Set: An ontology and human-labeled dataset for audio events", ICASSP. |
| `audioset_aliases.yaml` | This project: common free-text labels mapped onto AudioSet class names | MIT (this repository) |
| `musicbrainz_genres.txt` | MusicBrainz genre list, https://musicbrainz.org/genres (`/ws/2/genre/all`) | CC0 (MusicBrainz core data) |
| `discogs_styles.txt` | The names of the 400 most common Discogs styles, in `Genre---Style` form, as used by the MTG Essentia `genre_discogs400` classifier. Only the style names are included (no model files or weights). | Style names are Discogs' public taxonomy; see https://www.discogs.com |
| `moods.yaml` | This project: mood terms and their placements on Russell's circumplex model (Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology, 39(6), 1161–1178). The coordinates are this project's own approximations. | MIT (this repository) |

The derived AudioSet data is distributed under CC BY-SA 4.0; if you modify and
redistribute `audioset.json`, keep the same licence and this attribution.
