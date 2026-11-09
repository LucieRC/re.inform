## Une enquête européenne, menée en France en 2018

Les profils partent de l'**European Values Study** (EVS), une enquête européenne de référence sur les valeurs et les opinions. Nous utilisons sa vague dite « 2017 », réalisée en France **du 2 mars au 16 août 2018**, en face à face avec un enquêteur.

**1 870 adultes** de 18 ans et plus y ont répondu.

Dans un sondage, certaines catégories de personnes sont toujours un peu trop ou pas assez nombreuses. D'où le **redressement** : l'enquête fournit un « poids » pour chaque personne, qui compte un peu plus ou un peu moins afin que l'échantillon ressemble à la population française selon l'âge, le sexe, le niveau d'études et la région. Tous nos pourcentages utilisent ces poids.

## Huit dimensions pour comparer les opinions

Plutôt que de comparer des centaines de réponses une à une, nous avons construit huit dimensions, choisies parce qu'elles comptent dans une conversation. Chacune résume une ou plusieurs questions de l'enquête.

1. **Environnement ou croissance :** ce qui doit passer en premier, et la confiance accordée au mouvement écologiste.
2. **Confiance dans les institutions :** presse, Parlement, administration, gouvernement, partis, justice, Union européenne, ONU ; satisfaction envers le système politique ; sentiment que les élections sont honnêtes.
3. **Autorité et ordre :** envie d'un chef fort ou d'un régime militaire, respect de l'autorité, obéissance des enfants, peine de mort, droit de l'État à surveiller la population.
4. **Ouverture :** effet attribué à l'immigration, priorité à l'emploi des nationaux, immigrés comme voisins, sentiment d'être proche du monde.
5. **Gauche ou droite :** la position que la personne se donne elle-même.
6. **Engagement politique :** intérêt pour la politique, pétitions, boycotts, manifestations.
7. **Difficultés de vie :** revenus, satisfaction dans la vie, sentiment d'avoir prise sur sa vie.
8. **Tradition :** importance de Dieu, opinion sur l'homosexualité, l'avortement, l'euthanasie et l'homoparentalité.

Une personne qui n'a répondu à aucune question d'une dimension reçoit la valeur du milieu de l'échantillon (la **médiane**). C'est surtout le cas pour la position gauche-droite, sans réponse chez 17,3 % des personnes. Pour la dimension environnement, c'est 1,7 %, et presque personne pour les autres.

## Des groupes formés par le calcul

Imaginez chaque personne placée sur une carte à huit directions, une par dimension. Le calcul cherche six points de ralliement, puis rattache chaque personne au plus proche, de sorte que les personnes d'un même groupe soient aussi proches que possible les unes des autres. Personne n'a décidé à l'avance qui irait où.

**Côté technique :** un **k-means** (méthode classique de regroupement automatique) sur les huit dimensions, mises à la même échelle (**standardisation**) et en tenant compte des poids. Le résultat dépend du point de départ : le calcul est donc lancé 50 fois et le meilleur essai est conservé. Le point de départ est fixé (**graine aléatoire** 42), si bien que le même code redonne le même résultat.

## Six groupes : un choix, pas une évidence

Le calcul ne dit pas combien de groupes il faut : c'est à nous de le choisir. Nous avons comparé de 5 à 9 groupes, avec trois indicateurs.

- **La silhouette :** les groupes sont-ils bien séparés ? (1 : parfaitement ; 0 : ils se confondent.)
- **La stabilité :** retrouve-t-on les mêmes groupes en relançant le calcul depuis un autre point de départ ? (1 : à l'identique ; 0 : pas mieux que le hasard.)
- **Le plus petit groupe :** la part de la population qu'il représente.

Résultats :

- **5 groupes :** silhouette 0,124 ; stabilité 0,92 ; plus petit groupe 12,7 %.
- **6 groupes :** silhouette 0,116 ; stabilité 0,75 ; plus petit groupe 13,2 %.
- **7 groupes :** silhouette 0,114 ; stabilité 0,48 ; plus petit groupe 10,4 %.
- **8 groupes :** silhouette 0,111 ; stabilité 0,69 ; plus petit groupe 8,3 %.
- **9 groupes :** silhouette 0,112 ; stabilité 0,78 ; plus petit groupe 8,0 %.

**Ces chiffres ne désignent pas six comme la bonne réponse.** Cinq groupes font même mieux en silhouette et en stabilité. Partout, la silhouette est faible, ce qui signifie que les groupes se chevauchent : les opinions des gens forment un continuum, sans frontières nettes.

Six groupes ont été retenus pour des raisons pratiques :

- c'est un nombre facile à retenir ;
- chaque groupe reste important (13 % ou plus) ;
- c'est à partir de six qu'apparaît un groupe utile pour re.inform : des personnes attachées à la planète mais en colère contre « le système », un public qui nous semble particulièrement exposé aux récits complotistes.

## Des personnages attribués à la main

Associer un personnage à un groupe **n'est pas un calcul : c'est un choix humain**. Nous avons lu le profil de chaque groupe et nous l'avons rapproché des personnages caricaturaux, dans l'esprit du média Bon Pote, déjà présents sur le site.

- Cinq groupes ont reçu l'un des cinq personnages existants.
- Le sixième ne correspondait à aucun : nous avons créé **Max Onnousment**.
- Quand les données contredisaient un personnage, nous avons changé le personnage. Catherine de France, par exemple, est désormais attachée à la protection de la nature : 68 % des personnes de son groupe font passer l'environnement avant la croissance.

Les noms, les phrases types et les conseils pour parler à chaque profil **ne viennent pas de l'enquête**. Ils ont été rédigés à la main, à partir du profil de chaque groupe et des personnages d'origine. L'enquête n'a jamais testé ce qui convainc ces personnes.

## Des parts à lire comme des ordres de grandeur

Chaque pourcentage indique le poids estimé du groupe parmi les adultes en France en 2018. Le total fait 99 % parce que les chiffres sont arrondis. Chaque groupe repose sur 225 à 505 personnes interrogées.

L'incertitude se mesure de deux façons, qui ne disent pas la même chose :

- **Environ 2 points** en plus ou en moins, pour chaque groupe, **si l'on considère les groupes comme acquis**. La seule erreur vient alors du fait qu'on a interrogé un échantillon et non toute la population.
- **De 3 à 8 points** selon le groupe **si l'on refait tout le regroupement** sur des échantillons tirés au hasard parmi les répondants (500 fois, méthode dite du **bootstrap**). Les frontières entre groupes bougent alors aussi.

Avec cette seconde méthode, au niveau de confiance habituel de 95 %, la part de chaque groupe se situe dans ces fourchettes :

- **David Rassuriste :** 13 à 28 %
- **Roxane Bobo :** 12 à 25 %
- **Max Onnousment :** 12 à 27 %
- **Claire Céfoutu :** 11 à 19 %
- **Catherine de France :** 11 à 18 %
- **Jean-Michel Cyclesolaire :** 9 à 17 %

Ces fourchettes sont larges et ne sont pas centrées sur le pourcentage affiché. Mieux vaut donc retenir « environ un quart » ou « environ un sur sept ». Les écarts entre les groupes proches, comme 14 % et 13 %, ne veulent rien dire.

## Les limites à garder en tête

- **Les données datent de 2018.** Elles sont antérieures à la pandémie de Covid, à la guerre en Ukraine et à l'inflation qui a suivi. Les opinions et le poids des groupes ont pu changer.
- **Le climat est à peine mesuré.** L'enquête ne pose aucune question sur le climat lui-même. La plus proche demande s'il faut donner la priorité à la protection de l'environnement ou à la croissance et à l'emploi. La seule autre information utilisée est la confiance dans le mouvement écologiste.
- **Le côté « climat » des personnages est donc approximatif.** L'enquête ne dit pas qui doute du réchauffement ni qui croit que la technologie réglera tout. Dans le groupe de Jean-Michel, « l'oncle climatosceptique », 26 % des personnes font d'ailleurs passer l'environnement avant la croissance.
- **Les groupes sont des simplifications.** Ils se chevauchent, et beaucoup de gens se trouvent à la frontière de deux groupes. D'autres choix (d'autres dimensions, un autre nombre de groupes) donneraient un découpage différent.
- **Les personnages sont des caricatures.** Aucun profil ne décrit une personne réelle : il résume des tendances moyennes. Il aide à préparer une conversation, mais ne vous dit pas ce que pense la personne en face de vous. Le mieux reste de l'écouter.

## Sources et code

- [European Values Study](https://europeanvaluesstudy.eu), site de l'enquête.
- [Données : Joint EVS/WVS 2017-2022, version 5.0.0](https://doi.org/10.4232/1.14320), archives GESIS, dont nous avons extrait la France.
- [Données source : EVS 2017, version 5.0.0](https://doi.org/10.4232/1.13897).
- [Documentation sur les poids de l'enquête EVS 2017](https://doi.org/10.21241/ssoar.70113).
- [Bon Pote](https://bonpote.com), média dont l'esprit a inspiré les personnages.

**Le code est public**, dans [`EVS/bin`](https://github.com/LucieRC/re.inform/tree/main/EVS/bin). Trois scripts produisent tous les résultats chiffrés : `stance_personas.py` construit les groupes, `stance_margins.py` calcule les marges d'erreur et `export_personas.py` associe les groupes aux personnages. Les données brutes se téléchargent auprès de GESIS, après inscription.
