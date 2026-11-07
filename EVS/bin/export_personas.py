#!/usr/bin/env python3
"""Write the chosen persona list (option A, 6 personas) into the website's persona store.

Decision (owner, 2026-11-07): list A from docs/proposals/personas.html, Bon Pote-style names.

What it does
- Reads the 6 stance clusters from output/fr_stance_k6.json (built by bin/stance_personas.py,
  random_state 42) and the crosswalk output/fr_stance_k6_vs_values_k10.csv.
- Updates the website store website2/backend/data/persona_archetypes.json:
  * the 5 Bon Pote personas keep their ids (1..5), so every answer that references them still
    resolves; their population share, EVS profile and the descriptions the data contradicts are
    updated (Catherine, Claire);
  * one new persona is added (Max Onnousment), with the next free id;
  * the 10 old EVS groups (01..10) are kept as retired aliases: same ids, same layer
    ("evs_cluster", which the backend refuses as an answer target), plus `alias_of` = the new
    persona that receives the largest part of that old group.
- Backs up the target file first (persona_archetypes.json.bak-YYYYmmdd-HHMMSS).
- Checks that every persona_archetype_id used in argumentation.json resolves.

Deterministic: no randomness here; the clustering seed lives in stance_personas.py. Re-running
gives the same content (only `updated_at` changes).

Run from EVS/: python bin/export_personas.py [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "website2" / "backend" / "data" / "persona_archetypes.json"
DEFAULT_ARGS = ROOT / "website2" / "backend" / "data" / "argumentation.json"

LIST_ID = "A-2026-11"
SOURCE = {
    "project": "EVS",
    "source_file": "EVS/output/fr_stance_k6.json",
    "script": "EVS/bin/stance_personas.py (random_state 42)",
    "survey": "European Values Study 2017-2018, France, n=1870, weights gwght",
}

# stance cluster -> persona. `id` None means a new persona (gets the next free id).
# `set` replaces fields; nested dicts given here replace the whole nested value.
PERSONAS: Dict[str, Dict[str, Any]] = {
    "S01": {"id": 3, "destin_commun_family": "Libéraux optimistes", "set": {}},
    "S02": {
        "id": 5,
        "destin_commun_family": "Militants désabusés",
        "set": {  # data: environment first, trusts science/institutions, left, politically active
            "description": "Signe les pétitions et va aux marches climat",
            "short_label": "La convaincue engagée",
            "value_system": "Sensibilité écologique et sens de la justice ; fait confiance à la science. Déjà "
                            "engagée : signe des pétitions, manifeste, vote à gauche ou écologiste.",
            "dimensions": {
                "epistemic_style": {"score": -2, "label": "Fait confiance à la science et aux données (GIEC, chercheurs)"},
                "moral_foundations": {"score": -2, "label": "Soin et justice : protéger les plus fragiles et le vivant"},
                "threat_sensitivity": {"score": -2, "label": "Ouverte au changement, elle le souhaite même"},
            },
            "engagement_stance": "Engagée",
            "social_anchoring": "Associations, collectifs militants, amis engagés, médias de gauche et écologistes.",
            "persuasion": {
                "approach": "Déjà convaincue : l'enjeu n'est pas de la convaincre mais de l'outiller. Lui donner des "
                            "réponses sourcées et des façons de parler aux autres profils sans les braquer, et l'aider "
                            "à garder espoir sans s'enfermer dans l'entre-soi.",
                "format": "calm_discussion_evidence",
                "do": [
                    "Lui donner des sources solides à relayer",
                    "L'aider à adapter son discours à la personne en face",
                    "Valoriser l'action collective qu'elle mène déjà",
                ],
                "dont": [
                    "Lui prêcher ce qu'elle sait déjà",
                    "La réduire à des écogestes ou à un style de vie",
                    "Encourager un ton moralisateur envers les autres",
                ],
                "messengers": [
                    "Un chercheur ou un vulgarisateur scientifique",
                    "Une association reconnue",
                    "Un militant expérimenté",
                ],
            },
            "typical_phrases": [
                "On a toutes les études, il faut agir maintenant",
                "Il faut changer le système, pas seulement nos gestes",
                "J'ai signé la pétition, tu viens à la marche ?",
            ],
        },
    },
    "S03": {
        "id": None,
        "destin_commun_family": "Laissés pour compte",
        "set": {
            "name": "Max Onnousment",
            "description": "Le climat, oui, mais on nous ment",
            "short_label": "Le révolté anti-système",
            "icon_emoji": "🧢",
            "accent_color": "#B4532A",
            "value_system": "Sens aigu de la justice et colère contre « le système » : médias, gouvernement "
                            "et grandes entreprises sont soupçonnés de mentir ou de servir les puissants. "
                            "Tient à la planète, mais ne fait pas confiance à ceux qui en parlent.",
            "dimensions": {
                "epistemic_style": {"score": 1, "label": "Se méfie des médias et des institutions ; croit plus volontiers les sources alternatives"},
                "moral_foundations": {"score": -2, "label": "Justice et équité : ce sont toujours les mêmes qui paient"},
                "threat_sensitivity": {"score": -1, "label": "Veut que ça change, mais pas par ceux d'en haut"},
            },
            "engagement_stance": "Concerné mais méfiant",
            "social_anchoring": "Amis, collègues, collectifs de terrain, créateurs de contenu indépendants ; pas les médias « mainstream ».",
            "persuasion": {
                "approach": "Partir de sa colère contre les injustices au lieu de la contredire. Montrer comment se "
                            "construit une preuve (qui finance, qui vérifie, qui contredit) plutôt que de demander de "
                            "faire confiance aux institutions, et faire remarquer que les récits « on nous ment » "
                            "profitent souvent aux mêmes puissants qu'il dénonce.",
                "format": "moral_care_fairness",
                "do": [
                    "Reconnaître les vraies injustices et les vrais scandales",
                    "Montrer les sources et la méthode, pas seulement l'autorité",
                    "Relier l'action climatique à la justice sociale",
                ],
                "dont": [
                    "Le traiter de complotiste",
                    "Défendre en bloc le gouvernement ou les médias",
                    "Le renvoyer à ses petits gestes individuels",
                ],
                "messengers": [
                    "Un syndicaliste ou un militant de terrain",
                    "Un journaliste d'investigation indépendant",
                    "Un pair qui a lui-même changé d'avis",
                ],
            },
            "typical_phrases": [
                "On nous ment depuis le début",
                "C'est aux gros pollueurs de payer",
                "Les médias sont tous achetés",
            ],
            "layer": "archetype",
            "imported_from": {"project": "re.inform", "source_file": "EVS/bin/export_personas.py"},
        },
    },
    "S04": {
        "id": 4,
        "destin_commun_family": "Attentistes",
        "set": {
            "description": "De toute façon, ça ne changera rien",
            "short_label": "La désabusée",
            "value_system": "Ne fait plus confiance à personne et ne s'intéresse plus à la politique ; "
                            "l'emploi et le quotidien passent avant la planète.",
            "dimensions": {
                "epistemic_style": {"score": 1, "label": "Ne croit plus grand monde, ni les médias ni les politiques"},
                "moral_foundations": {"score": 0, "label": "Ni militante ni traditionaliste : surtout lasse"},
                "threat_sensitivity": {"score": 1, "label": "N'attend plus rien du changement ; se protège"},
            },
            "engagement_stance": "Désengagée",
            "social_anchoring": "Son entourage proche, ses collègues ; plus aucun parti.",
            "persuasion": {
                "approach": "Ne pas commencer par le climat ni par la politique. Partir de son quotidien (factures, "
                            "santé, emploi, sa région) et montrer des gains concrets et proches, portés par des gens "
                            "qu'elle connaît. Lui redonner un peu de prise sur les choses plutôt qu'ajouter un sujet "
                            "d'angoisse.",
                "format": "example_social_proof",
                "do": [
                    "Parler concret et local : factures, santé, emploi",
                    "Montrer des gens comme elle qui y ont gagné quelque chose",
                    "Proposer un petit pas facile, sans engagement politique",
                ],
                "dont": [
                    "Lui faire la morale ou la culpabiliser",
                    "Lancer un débat politique ou partisan",
                    "La noyer sous les chiffres et les catastrophes",
                ],
                "messengers": [
                    "Un voisin, un collègue",
                    "Un professionnel de proximité : médecin, artisan, maire",
                ],
            },
            "typical_phrases": [
                "De toute façon c'est trop tard",
                "Tous pourris, ça ne sert à rien",
                "J'ai d'autres soucis à la fin du mois",
            ],
        },
    },
    "S05": {
        "id": 2,
        "destin_commun_family": "Stabilisateurs",
        "set": {
            "value_system": "Foi, tradition et transmission ; confiance dans les institutions. Tient à protéger "
                            "la nature, qu'elle voit comme un héritage à transmettre.",
            "dimensions": {
                "epistemic_style": {"score": -1, "label": "Fait confiance aux institutions établies et à l'Église"},
                "moral_foundations": {"score": 2, "label": "Loyauté, autorité, tradition, respect de l'ordre"},
                "threat_sensitivity": {"score": 1, "label": "Tient à ses repères, et veut protéger ce qui a de la valeur"},
            },
            "engagement_stance": "Concernée",
            "typical_phrases": [
                "De mon temps on ne gaspillait rien",
                "Il faut respecter l'ordre des choses",
                "Il faut prendre soin de la Création",
            ],
        },
    },
    "S06": {"id": 1, "destin_commun_family": "Identitaires", "set": {}},
}

ONE_LINERS = {  # plain-language summary used on the proposal page (kept for reference / methodology page)
    "S01": "Va bien, fait confiance au système, centre-droit ; le climat compte, mais ce n'est pas urgent.",
    "S02": "Déjà convaincue : l'environnement d'abord, fait confiance à la science, à gauche, signe et manifeste.",
    "S03": "En colère contre « le système » : tient à la planète mais se méfie des médias, du gouvernement et des élections.",
    "S04": "A décroché : ne fait confiance à personne, ne s'intéresse pas à la politique, l'emploi avant la planète.",
    "S05": "Foi et tradition ; fait confiance aux institutions et met souvent la nature en premier.",
    "S06": "Le « non » en colère : veut de l'ordre, se méfie des immigrés et des médias, l'environnement en dernier.",
}


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--target", default=str(DEFAULT_TARGET))
    ap.add_argument("--argumentation", default=str(DEFAULT_ARGS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    target = Path(args.target)
    stance = load_json(outdir / "fr_stance_k6.json")
    assert stance["k"] == 6 and stance["random_state"] == 42, "unexpected stance output"
    clusters = {c["stance_id"]: c for c in stance["clusters"]}

    # old EVS group (P01..P10) -> stance cluster receiving most of its people
    old_to_sid: Dict[str, str] = {}
    with open(outdir / "fr_stance_k6_vs_values_k10.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.pop(next(iter(row)))
            old_to_sid[pid] = max(row, key=lambda k: float(row[k]))

    store = load_json(target)
    personas: List[Dict[str, Any]] = store["persona_archetypes"]
    by_id = {p["id"]: p for p in personas}
    used_ids = {p["id"] for p in personas}

    # Resolve ids (new personas: reuse an earlier export by name, else next free id).
    sid_to_id: Dict[str, int] = {}
    for sid, spec in PERSONAS.items():
        if spec["id"] is not None:
            sid_to_id[sid] = spec["id"]
            continue
        name = spec["set"]["name"]
        existing = [p["id"] for p in personas if p.get("name") == name]
        sid_to_id[sid] = existing[0] if existing else max(used_ids) + 1
        used_ids.add(sid_to_id[sid])

    now = datetime.now(timezone.utc).isoformat()
    for sid, spec in PERSONAS.items():
        pid = sid_to_id[sid]
        c = clusters[sid]
        p = copy.deepcopy(by_id.get(pid, {"id": pid, "created_at": now}))
        p.update(copy.deepcopy(spec["set"]))
        p["layer"] = "archetype"
        p["persona_list"] = LIST_ID
        p["population_share_w"] = c["population_share_w"]
        p["destin_commun_family"] = spec["destin_commun_family"]
        p["summary"] = ONE_LINERS[sid]
        p["evs_profile"] = {
            **SOURCE,
            "stance_id": sid,
            "n_respondents": c["n_respondents"],
            "axis_z": {k[2:]: round(v, 3) for k, v in c.items() if k.startswith("z_")},
            "top_axes": c["top_axes"].split(";"),
        }
        p["aliases"] = []  # filled below
        p.pop("alias_of", None)
        p.pop("status", None)
        by_id[pid] = p

    # Old EVS groups become retired aliases.
    for p in list(by_id.values()):
        code = p.get("persona_code")
        if p.get("layer") == "evs_cluster" and code in old_to_sid:
            new_id = sid_to_id[old_to_sid[code]]
            p["status"] = "retired"
            p["alias_of"] = new_id
            p["retired_note"] = ("Old EVS value group (k=10). Replaced by persona list " + LIST_ID +
                                 "; most of its people are in persona " + str(new_id) + ".")
            by_id[new_id]["aliases"].append(p["id"])
    for sid in PERSONAS:
        by_id[sid_to_id[sid]]["aliases"].sort()

    new_list = [by_id[i] for i in sorted(by_id)]

    # Every referenced persona id must resolve to a current persona (directly or via alias_of).
    argstore = load_json(Path(args.argumentation))
    final = {p["id"]: p for p in new_list}
    used = sorted({ca.get("persona_archetype_id") for ca in argstore.get("counter_arguments", [])
                   if ca.get("persona_archetype_id") is not None})
    unresolved = []
    for u in used:
        p = final.get(u)
        tgt = final.get(p.get("alias_of")) if p and p.get("alias_of") else p
        if not tgt or tgt.get("layer") != "archetype":
            unresolved.append(u)
    print("persona ids used in argumentation.json:", used,
          "-> all resolve" if not unresolved else f"-> UNRESOLVED {unresolved}")
    if unresolved:
        return 1

    for sid in PERSONAS:
        p = final[sid_to_id[sid]]
        print(f"{sid} -> id {p['id']:>2} {p['name']:<26} {p['population_share_w']*100:4.1f}%  aliases {p['aliases']}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(target.name + f".bak-{stamp}")
    shutil.copy2(target, backup)
    store["updated_at"] = now
    store["persona_archetypes"] = new_list
    target.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    print("backup:", backup)
    print("wrote:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
