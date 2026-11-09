// ========================================
// re.inform - Interface text (i18n)
// ========================================
//
// Every piece of interface text lives in STRINGS below, grouped by page/area.
// The site is in French only for now. To add a language later: copy the `fr`
// block to e.g. `en`, translate the values, and set LANG (no page needs editing).
//
// In HTML:  <h1 data-i18n="home.title">…</h1>             → text
//           <p data-i18n-html="contribute.accountNotice">  → trusted markup from the catalog
//           data-i18n-placeholder / -title / -aria-label / -content="key" → that attribute
//   The French text is also written in the HTML itself, so pages read correctly
//   without JavaScript and with no flash; on load the catalog wins.
// In JS:    t('card.answers', { n: 3 })  → "3 réponses"
//   - `{name}` placeholders are filled from vars (numbers get French formatting).
//   - A value can be { one, other }: the form is picked from vars.n (French rules:
//     0 and 1 are singular).
//   - Keys ending in `Html` contain markup: escape any user text you pass in.
//
// Load this file first on every page (before theme.js and main.js).

const LANG = 'fr';
const LOCALE = 'fr-FR';

const STRINGS = {
    fr: {
        // ---- Shared ----
        'site.tagline': 'Renouer le dialogue, preuves à l’appui.',
        'site.pageTitle': '{page} - re.inform',
        'common.cancel': 'Annuler',
        'common.close': 'Fermer',
        'common.optional': '(facultatif)',
        'common.unknown': 'Inconnu',
        'common.untitled': 'Sans titre',
        'common.source': 'Source',
        'common.viewSource': 'Voir la source',
        'common.by': 'Par {name}',
        'common.createAccount': 'Créer un compte',
        'common.createFreeAccount': 'Créer un compte gratuit',
        'common.login': 'Se connecter',
        'common.loggingIn': 'Connexion…',
        'common.loading': 'Chargement…',
        'common.networkError': 'Problème de connexion. Veuillez réessayer.',
        'common.serverUnreachable': 'Impossible de joindre le serveur. Veuillez réessayer.',
        'common.redirecting': 'Redirection…',
        'common.redirectHome': 'Redirection vers l’accueil.',
        'common.redirectHomeHtml': 'Redirection vers <a href="index.html">l’accueil</a>…',
        'common.redirectHelp': 'Redirection vers « Contribuer ».',
        'common.redirectHelpHtml': 'Redirection vers <a href="index.html?mode=help">« Contribuer »</a>…',
        'common.redirectArticles': 'Redirection vers les articles.',
        'common.redirectArticlesHtml': 'Redirection vers <a href="resources.html">les articles</a>…',

        // ---- Unknown page (404.html) ----
        'notFound.metaTitle': 'Page introuvable - re.inform',
        'notFound.title': 'Page introuvable',
        'notFound.lead': 'Cette page n’existe pas ou a été déplacée.',
        'notFound.home': 'Retour à l’accueil',

        // ---- Header (same on every page) ----
        'nav.main': 'Navigation principale',
        'nav.debunk': 'Débunker',
        'nav.personas': 'Profils',
        'nav.resources': 'Ressources',
        'nav.resourcesLabel': 'Ressources',
        'nav.library': 'Bibliothèque',
        'nav.articles': 'Articles',
        'nav.searchPlaceholder': 'Rechercher une idée reçue…',
        'nav.searchLabel': 'Rechercher une idée reçue',
        'nav.searchPlaceholderShort': 'Rechercher…',
        'nav.menu': 'Menu',
        'nav.login': 'Connexion',
        'nav.logout': 'Déconnexion',
        'nav.profile': 'Votre profil ({name})',
        'nav.profileFallback': 'Profil',
        'theme.toggle': 'Changer de thème',
        'theme.toDark': 'Passer en mode sombre',
        'theme.toLight': 'Passer en mode clair',

        // ---- Time ----
        'time.justNow': 'à l’instant',

        // ---- Home: claims feed ----
        'home.metaTitle': 're.inform - Renouer le dialogue, preuves à l’appui',
        'home.metaDescription': 'Une plateforme collaborative pour répondre aux idées reçues sur le climat, preuves à l’appui, avec des réponses adaptées à la personne à qui vous parlez.',
        'home.title': 'Les idées reçues que vous entendrez, et comment y répondre.',
        'home.modeLabel': 'Que voulez-vous faire ?',
        'home.modeFind': 'Trouver une réponse',
        'home.modeHelp': 'Contribuer',
        'home.addClaim': 'Ajouter une idée reçue',
        'home.addResource': 'Ajouter une ressource',
        'home.statsLabel': 'Débunker en chiffres',
        'home.statClaims': { one: 'idée reçue', other: 'idées reçues' },
        'home.statAnswers': { one: 'réponse', other: 'réponses' },
        'home.statProfiles': { one: 'profil', other: 'profils' },
        'home.who': 'À qui parlez-vous ?',
        'home.topicsLabel': 'Filtrer par thème',
        'home.loading': 'Chargement des idées reçues…',
        'home.empty': 'Aucune idée reçue ne correspond à votre recherche.',
        'home.emptyPersona': 'Pas encore de réponse pour {names}.',
        'home.emptyPersonaHint': 'Vous savez comment lui parler ? Écrivez la première réponse.',
        'home.emptyPersonaCta': 'Contribuer : écrire une réponse',
        'home.or': ' ou ',
        'home.more': 'Voir plus',
        'feed.everyone': 'Tout le monde',
        'feed.allTopics': 'Tous les thèmes',
        'feed.clearAll': 'Tout effacer',
        'feed.loadError': 'Impossible de charger les idées reçues. Veuillez réessayer plus tard.',
        // Home, "Help out" mode: tasks and their cards
        'help.tasksLabel': 'Choisissez comment contribuer',
        'help.count': { one: '{n} idée reçue', other: '{n} idées reçues' },
        'help.vote.title': 'Évaluer l’urgence',
        'help.vote.sub': 'Un clic : faut-il y répondre en priorité ?',
        'help.vote.none': 'Personne n’a encore voté',
        'help.write.title': 'Écrire une réponse',
        'help.write.sub': 'Pour une personne qui n’a pas encore de réponse adaptée.',
        'help.write.signal': 'Pas encore de réponse pour {names}',
        'help.write.cta': 'Écrire pour {name}',
        'help.rate.title': 'Noter une réponse',
        'help.rate.sub': 'Rapide : cette réponse convaincrait-elle ?',
        'help.rate.signal': 'Réponse jamais notée',
        'help.rate.cta': 'Noter cette réponse',
        'help.empty': 'Rien à faire ici pour l’instant. Merci !',
        'help.more': 'Plus de façons de contribuer →',
        'help.moreGuest': 'Avec un compte gratuit, vous pouvez aussi écrire des réponses, partager des sources et suivre vos envois.',
        'card.howToAnswer': 'Comment répondre ?',
        'card.noAnswer': 'Pas encore de réponse',
        'card.beFirst': 'Écrivez la première →',
        'card.answers': { one: '{n} réponse', other: '{n} réponses' },

        // ---- Adding a claim (home + contribute) ----
        'claimForm.label': 'Qu’avez-vous entendu ?',
        'claimForm.help': 'Au repas de famille, en ligne, à la télé… Écrivez-la comme on l’entend. Pas besoin de source ni de réponse : d’autres vous aideront.',
        'claimForm.placeholder': 'Ex. : « Le climat a toujours changé, c’est naturel »',
        'claimForm.submit': 'Ajouter cette idée reçue',
        'claimForm.needAccountHtml': 'Il vous faut un compte gratuit pour publier. <a href="{registerHref}">Créez-en un</a> ou <a href="{loginHref}">connectez-vous</a> : vous reviendrez ici.',
        'claimForm.loginFirstHtml': 'Veuillez d’abord <a href="{registerHref}">créer un compte gratuit</a> ou <a href="{loginHref}">vous connecter</a>.',
        'claimForm.error': 'Impossible d’ajouter cette idée reçue.',
        'claimForm.successHtml': 'Merci, votre idée reçue est en ligne. <a href="{href}">La voir</a> ou <a href="{writeHref}">écrire une première réponse →</a>',

        // ---- Votes cast before signing up ----
        'votes.pendingHtml': {
            one: 'Vote enregistré. <a href="{href}">Créez un compte gratuit</a> et votre vote comptera.',
            other: 'Vote enregistré. <a href="{href}">Créez un compte gratuit</a> et vos {n} votes compteront.',
        },
        'votes.pendingToast': {
            one: 'Vote enregistré. Créez un compte gratuit et votre vote comptera.',
            other: 'Vote enregistré. Créez un compte gratuit et vos {n} votes compteront.',
        },
        'votes.claimed': { one: ' Votre vote compte maintenant.', other: ' Vos {n} votes comptent maintenant.' },
        'votes.waitingLogin': {
            one: 'Vous avez 1 vote en attente : il comptera dès que vous vous connecterez.',
            other: 'Vous avez {n} votes en attente : ils compteront dès que vous vous connecterez.',
        },
        'votes.waitingRegister': {
            one: 'Vous avez 1 vote en attente : il comptera dès votre inscription.',
            other: 'Vous avez {n} votes en attente : ils compteront dès votre inscription.',
        },

        // ---- Login / register ----
        'auth.username': 'Nom d’utilisateur',
        'auth.password': 'Mot de passe',
        'login.metaTitle': 'Connexion - re.inform',
        'login.metaDescription': 'Connectez-vous à votre compte re.inform.',
        'login.title': 'Connexion à re.inform',
        'login.lead': 'Retrouvez vos contributions et suivez votre progression.',
        'login.usernamePlaceholder': 'Votre nom d’utilisateur',
        'login.passwordPlaceholder': 'Votre mot de passe',
        'login.remember': 'Se souvenir de moi',
        'login.noAccount': 'Pas encore de compte ?',
        'login.registerLink': 'Inscrivez-vous',
        'login.success': 'Connexion réussie !{claimed} Redirection…',
        'login.failed': 'La connexion a échoué.',
        'register.metaTitle': 'Inscription - re.inform',
        'register.metaDescription': 'Créez votre compte re.inform.',
        'register.title': 'Créez votre compte',
        'register.lead': 'Rejoignez re.inform pour suivre vos contributions, gagner en réputation et aider à répondre aux idées reçues, preuves à l’appui.',
        'register.username': 'Nom d’utilisateur *',
        'register.usernameHint': '3 à 30 caractères : lettres (accents compris), chiffres, point, tiret ou tiret bas. Visible publiquement.',
        'register.usernamePlaceholder': 'Choisissez un nom d’utilisateur',
        'register.email': 'E-mail *',
        'register.emailHint': 'Votre e-mail ne sera pas visible publiquement.',
        'register.emailPlaceholder': 'vous@exemple.fr',
        'register.password': 'Mot de passe *',
        'register.passwordHint': 'Au moins 6 caractères.',
        'register.passwordPlaceholder': 'Choisissez un mot de passe sûr',
        'register.confirm': 'Confirmez le mot de passe *',
        'register.confirmPlaceholder': 'Le même mot de passe',
        'register.submit': 'Créer mon compte',
        'register.submitting': 'Création du compte…',
        'register.haveAccount': 'Vous avez déjà un compte ?',
        'register.loginLink': 'Connectez-vous',
        'register.mismatch': 'Les mots de passe ne correspondent pas.',
        'register.invalidEmail': 'Vérifiez l’adresse e-mail (par exemple vous@exemple.fr).',
        'register.success': 'Compte créé !{claimed} Redirection…',
        'register.failed': 'L’inscription a échoué.',

        // ---- Profile ----
        'profile.metaTitle': 'Profil - re.inform',
        'profile.metaDescription': 'Votre profil re.inform et vos contributions.',
        'profile.changePhoto': 'Changer de photo',
        'profile.uploading': 'Envoi…',
        'profile.uploadFailed': 'L’envoi a échoué.',
        'profile.uploadRetry': 'L’envoi a échoué. Veuillez réessayer.',
        'profile.follow': 'Suivre',
        'profile.unfollow': 'Ne plus suivre',
        'profile.notFound': 'Membre introuvable',
        'profile.memberSinceHtml': 'Membre depuis {date}',
        'profile.reputation': 'Réputation',
        'profile.contributions': 'Contributions',
        'profile.following': 'Abonnements',
        'profile.followers': 'Abonnés',
        'profile.statContributions': { one: 'Contribution', other: 'Contributions' },
        'profile.statFollowing': { one: 'Abonnement', other: 'Abonnements' },
        'profile.statFollowers': { one: 'Abonné', other: 'Abonnés' },
        'profile.notFoundHtml': 'Ce membre n’existe pas ou son compte a été supprimé. <a href="index.html">Retour à l’accueil</a>',
        'profile.rolesTitle': 'Vos rôles',
        'profile.rolesLoading': 'Chargement des rôles…',
        'profile.tagsTitle': 'Thèmes que vous suivez',
        'profile.tagSaveError': 'Impossible d’enregistrer ce thème. Veuillez réessayer.',
        'profile.tagsLoadError': 'Impossible de charger les thèmes.',
        'profile.followingTitle': 'Personnes que vous suivez',
        'profile.followingEmpty': 'Vous ne suivez encore personne.',
        'profile.followingEmptyLong': 'Vous ne suivez encore personne. Ouvrez la page d’un membre pour le suivre !',
        'profile.followingLoadError': 'Impossible de charger vos abonnements.',
        'profile.recoTitle': 'Recommandé pour vous',
        'profile.recoLead': 'Des idées reçues qui attendent une réponse, choisies selon les thèmes que vous suivez.',
        'profile.recoLoading': 'Recherche d’idées reçues pour vous…',
        'profile.queueTitle': 'Sources à vérifier',
        'profile.queueLead': 'Des réponses attendent un vérificateur. Ouvrez-en une, suivez ses liens et dites s’ils la confirment.',
        'profile.activityTitle': 'Du côté des personnes que vous suivez',
        'profile.activityLoading': 'Chargement de l’activité récente…',
        'profile.trackerTitle': 'Vos envois',
        'profile.trackerLead': 'Suivez vos envois pendant leur relecture.',
        'profile.trackerLoading': 'Chargement de vos envois…',
        'profile.endorsementsTitle': 'Soutiens reçus',
        'profile.endorsementsEmpty': 'Aucun soutien pour l’instant.',
        'profile.contributionsTitle': 'Contributions',
        'profile.contributionsMine': 'Vos contributions',
        'profile.contributionsLoading': 'Chargement des contributions…',
        'profile.contributionsEmpty': 'Aucune contribution pour l’instant.',
        'profile.contributionsEmptyMineHtml': 'Aucune contribution pour l’instant. <a href="index.html?mode=help">Écrivez votre première réponse !</a>',
        'profile.draftsTitle': 'Brouillons',
        'profile.draftFor': 'Pour {name}',
        'profile.draftResume': 'Reprendre',
        'profile.draftDelete': 'Supprimer ce brouillon',
        'profile.draftDeleted': 'Brouillon supprimé',
        'profile.draftDeleteError': 'Impossible de supprimer ce brouillon.',
        'profile.helpTitle': 'Comment contribuer',
        'profile.helpLead': 'Dites-nous ce qui vous décrit : nous vous proposons ce que vous pouvez faire tout de suite.',

        'userRole.registered': 'Membre',
        'userRole.member': 'Membre',
        'userRole.trusted': 'Membre de confiance',
        'userRole.moderator': 'Modérateur',
        'userRole.admin': 'Administrateur',

        // Level bar (reputation → progression)
        'level.observer': 'Observateur',
        'level.engager': 'Participant',
        'level.strategist': 'Stratège',
        'level.researcher': 'Chercheur',
        'level.leader': 'Pilier de la communauté',
        'level.progressLabel': 'Progression vers le niveau suivant',
        'level.next': { one: 'Plus que {n} pt avant {level}', other: 'Plus que {n} pts avant {level}' },
        'level.top': 'Niveau maximum atteint',
        'level.valueText': '{level}, {pct} % vers {next}',

        // Contributor roles (names and texts are keyed by the role id the API sends)
        'role.wordsmith.name': 'Plume',
        'role.wordsmith.duty': 'Écrire des réponses, et réécrire les bonnes pour un autre profil.',
        'role.wordsmith.power': 'Proposer des versions améliorées des réponses des autres.',
        'role.wordsmith.unit': 'réponses notées 3,5 ★ ou plus par 3 membres',
        'role.sourcer.name': 'Vérificateur',
        'role.sourcer.duty': 'Vérifier que les sources citées par une réponse l’appuient vraiment.',
        'role.sourcer.power': 'Attribuer le tampon « source vérifiée ». La vérification d’un vérificateur confirmé compte double.',
        'role.sourcer.unit': 'vérifications conformes au verdict final',
        'role.field_agent.name': 'Agent de terrain',
        'role.field_agent.duty': 'Essayer des réponses dans de vraies conversations et raconter ce qui s’est passé.',
        'role.field_agent.power': 'Les retours de terrain arrivent bientôt : vous serez parmi les premiers sollicités.',
        'role.tier.confirmed': 'Confirmé',
        'role.tier.apprentice': 'Apprenti',
        'role.chipTitle': '{name} · {tier}',
        'role.progressLabel': 'Progression vers {name} confirmé',
        'role.private': 'Ce rôle n’est visible que par vous.',
        'role.stepDown': 'Quitter ce rôle',
        'role.take': 'Prendre ce rôle',
        'role.taken': 'Rôle pris. C’est votre travail qui vous y confirmera.',
        'role.dropped': 'Vous avez quitté ce rôle. Votre travail passé comptera si vous revenez.',
        'role.error': 'Impossible de mettre à jour votre rôle.',
        'role.networkError': 'Problème de connexion pendant la mise à jour de votre rôle.',
        'role.loadError': 'Impossible de charger les rôles.',

        // Sources to check (Sourcer queue)
        'queue.empty': 'Rien à vérifier pour le moment. Merci !',
        'queue.loadError': 'Impossible de charger la liste.',
        'queue.checks': { one: '{n} vérification', other: '{n} vérifications' },
        'queue.noCheck': 'Aucune vérification',
        'queue.forPersona': 'Pour {name} · ',

        // Submission tracker
        'status.pending': 'En attente de relecture',
        'status.approved': 'Validé',
        'status.rejected': 'À retravailler',
        'tracker.source': 'Source',
        'tracker.response': 'Réponse',
        'tracker.claim': 'Idée reçue',
        'tracker.colType': 'Type',
        'tracker.colWhat': 'Contenu',
        'tracker.colSent': 'Envoyé le',
        'tracker.colStatus': 'Statut',
        'tracker.emptyHtml': 'Vous n’avez encore rien envoyé. <a href="index.html?mode=help">Répondez à une idée reçue ou partagez une source →</a>',
        'tracker.loadError': 'Impossible de charger vos envois pour le moment.',

        // Recommendations
        'reco.loadError': 'Impossible de charger les recommandations pour le moment.',
        'reco.pickTopics': 'Dites-nous quels thèmes vous intéressent : nous vous proposerons des idées reçues auxquelles répondre.',
        'reco.pickTopicsBtn': '+ Choisir des thèmes',
        'reco.noneHtml': 'Aucune idée reçue en attente dans vos thèmes pour le moment. Beau travail ! <a href="index.html">Voir toutes les idées reçues →</a>',
        'reco.because': 'Parce que vous suivez {tags}',
        'reco.noResponse': 'Pas encore de réponse',
        'reco.responses': { one: '{n} réponse', other: '{n} réponses' },
        'reco.beFirst': 'Écrivez la première réponse →',
        'reco.addYours': 'Ajoutez votre réponse →',

        // Activity from people you follow
        'activity.emptyHtml': 'Aucune activité récente des personnes que vous suivez. <a href="index.html">Trouver des personnes à suivre →</a>',
        'activity.loadError': 'Impossible de charger l’activité pour le moment.',
        'activity.answeredClaim': 'a répondu à une idée reçue :',
        'activity.wroteResponse': 'a écrit une réponse :',
        'activity.sharedSource': 'a partagé une source :',

        // Endorsements / contribution cards
        'contrib.quote': 'Citation :',
        'contrib.response': 'Réponse :',
        'contrib.title': 'Titre :',
        'contrib.anonymous': 'Contribution anonyme',
        'contrib.viewClaim': 'Voir l’idée reçue →',

        // ---- Contribute ----
        'contribute.metaTitle': 'Contribuer - re.inform',
        'contribute.metaDescription': 'Choisissez ce qui vous décrit et commencez à contribuer en quelques minutes : voter, répondre à une idée reçue, partager une source, dessiner ou coder.',
        'contribute.eyebrow': 'Contribuer',
        'contribute.title': 'Contribuez à votre façon',
        'contribute.lead': 'Aucune expertise requise. Dites-nous ce qui vous décrit et nous vous montrerons ce que vous pouvez faire tout de suite. La plupart des tâches prennent quelques minutes.',
        'contribute.accountNoticeHtml': '<strong>Vous pouvez voter tout de suite, sans compte.</strong> Pour écrire ou partager, il vous faut un compte gratuit : cela prend 30 secondes, et vous reviendrez directement ici.',
        'contribute.haveAccount': 'J’ai déjà un compte',
        'contribute.step1': 'Qu’est-ce qui vous décrit le mieux ?',
        'contribute.step2': 'Choisissez quoi faire',
        'contribute.prioritizeTitle': 'Quelles idées reçues entendez-vous le plus souvent ?',
        'contribute.prioritizeLead': 'Un clic par idée reçue. Les plus votées reçoivent une réponse en premier.',
        'contribute.answerTitle': 'Idées reçues en attente de réponse',
        'contribute.answerLead': 'Chaque idée reçue a besoin d’une réponse pour chaque type de personne. Choisissez une place libre : vous verrez des conseils sur cette personne pendant que vous écrivez. Quelques phrases suffisent.',
        'contribute.claimTitle': 'Ajoutez une idée reçue que vous avez entendue',
        'contribute.resourceTitle': 'Partagez une source ou un visuel',
        'contribute.resourceLead': 'Une étude, un article, un rapport, un graphique ou une infographie qui aide à répondre à une idée reçue. Une fois dans la bibliothèque, toute personne qui écrit une réponse pourra la citer.',
        'contribute.freeAccount': ' · compte gratuit',
        'contribute.needAccountHtml': 'Il vous faut un compte gratuit pour enregistrer ceci. <a href="{registerHref}">Créez-en un</a> ou <a href="{loginHref}">connectez-vous</a> : vous reviendrez exactement ici.',
        'contribute.voteNoAccountHtml': 'Pas besoin de compte pour voter. Vos votes sont enregistrés, et ils compteront dès que vous <a href="{registerHref}">créerez un compte gratuit</a>.',
        'contribute.gapsEmptyHtml': 'Toutes les idées reçues ont une réponse pour chaque profil. <a href="index.html">Parcourez-les</a> pour en améliorer une.',
        'contribute.gapFor': 'Pour {name}',
        'contribute.seeAll': 'Voir toutes les idées reçues →',

        // "What describes you" profiles
        'who.quick.title': 'J’ai quelques minutes',
        'who.quick.who': 'Tout le monde, sans expertise',
        'who.talk.title': 'J’en parle avec mon entourage',
        'who.talk.who': 'Famille, amis, collègues, en ligne',
        'who.science.title': 'Je suis à l’aise avec la science et les sources',
        'who.science.who': 'Recherche, études, vérification des faits',
        'who.write.title': 'J’écris, j’enseigne ou je communique',
        'who.write.who': 'Journalisme, enseignement, militantisme',
        'who.visual.title': 'Je fais du graphisme ou de l’illustration',
        'who.visual.who': 'Graphisme, illustration',
        'who.tech.title': 'Je code ou je travaille avec des données',
        'who.tech.who': 'Développement, science des données',

        // Tasks
        'task.prioritize.title': 'Votez pour les idées reçues que vous entendez le plus',
        'task.prioritize.desc': 'Un clic par idée reçue. Cela décide de ce qui reçoit une réponse en premier.',
        'task.prioritize.time': '1 min',
        'task.rate.title': 'Notez une réponse',
        'task.rate.desc': 'Lisez une réponse et dites si elle convaincrait la personne visée.',
        'task.rate.time': '30 s',
        'task.claim.title': 'Ajoutez une idée reçue que vous avez entendue',
        'task.claim.desc': 'Juste la phrase. Pas besoin de source ni de réponse.',
        'task.claim.time': '1 min',
        'task.answer.title': 'Répondez à une idée reçue pour un profil précis',
        'task.answer.desc': 'Choisissez une idée reçue et un profil encore sans réponse. Quelques phrases suffisent.',
        'task.answer.time': '5 min',
        'task.personas.title': 'Découvrez les profils',
        'task.personas.desc': 'Cinq types de personnes que vous pourriez croiser, et ce qui fait bouger chacune.',
        'task.personas.time': '5 min',
        'task.review.title': 'Vérifiez les réponses à une idée reçue',
        'task.review.desc': 'Contrôlez les sources, notez les réponses et proposez une version corrigée si quelque chose cloche.',
        'task.review.time': '10 min',
        'task.resource.title': 'Partagez une étude, un article ou un rapport',
        'task.resource.desc': 'Ajoutez des preuves à la bibliothèque commune pour que chacun puisse les citer.',
        'task.resource.time': '2 min',
        'task.improve.title': 'Rendez une réponse plus convaincante',
        'task.improve.desc': 'Ouvrez une idée reçue et utilisez « Améliorer » sur une réponse trop sèche ou trop moralisatrice.',
        'task.improve.time': '10 min',
        'task.visual.title': 'Transformez une réponse en visuel',
        'task.visual.desc': 'Une infographie circule plus loin qu’un texte. Créez-en une, puis partagez son lien dans la bibliothèque.',
        'task.visual.time': '1 h et plus',
        'task.code.title': 'Améliorez le site',
        'task.code.desc': 'Flask et JavaScript simple. Choisissez un ticket ouvert sur GitHub ou proposez le vôtre.',
        'task.code.time': 'À votre rythme',
        'task.data.title': 'Travaillez sur la recherche derrière les profils',
        'task.data.desc': 'Les profils sont construits à partir des données de l’enquête European Values Study, avec des scripts Python. Aidez à affiner la méthode.',
        'task.data.time': 'À votre rythme',

        // Claim priority votes
        'priority.low': 'Faible',
        'priority.lowHint': 'Rarement entendue',
        'priority.medium': 'Moyenne',
        'priority.mediumHint': 'Mérite une réponse',
        'priority.critical': 'Critique',
        'priority.criticalHint': 'À traiter en priorité',
        'priority.groupLabel': 'À quel point est-il urgent d’y répondre ?',
        'priority.tally': { one: '{n} vote · moyenne {avg}/5', other: '{n} votes · moyenne {avg}/5' },
        'priority.noRatings': 'Pas encore de vote',
        'priority.empty': 'Aucune idée reçue à évaluer pour l’instant.',
        'priority.error': 'Impossible d’enregistrer votre vote.',
        'priority.thanks': 'Merci, votre vote aide à fixer les priorités.',

        // Share a resource
        'resourceForm.url': 'Lien',
        'resourceForm.urlPlaceholder': 'https://exemple.fr/etude',
        'resourceForm.title': 'Titre',
        'resourceForm.titlePlaceholder': 'Rapport de synthèse du GIEC (AR6)',
        'resourceForm.description': 'Pourquoi c’est utile',
        'resourceForm.descriptionPlaceholder': 'Ce que ça montre, et quand s’en servir.',
        'resourceForm.claim': 'Idée reçue liée',
        'resourceForm.noClaim': 'Pas liée à une idée reçue précise',
        'resourceForm.submit': 'Partager',
        'resourceForm.titleOrNote': 'Titre ou courte note',
        'resourceForm.add': 'Ajouter cette ressource',
        'resourceForm.loginFirst': 'Connectez-vous pour partager une ressource.',
        'resourceForm.needLink': 'Ajoutez un lien.',
        'resourceForm.error': 'Impossible d’enregistrer la ressource.',
        'resourceForm.success': 'Merci ! Votre source apparaîtra dans la bibliothèque dès qu’un modérateur l’aura vérifiée.',

        // ---- Personas ----
        'personas.metaTitle': 'Profils - re.inform',
        'personas.metaDescription': 'Les profils des personnes à qui vous parlez, et leur part dans la population française.',
        'personas.title': 'Profils',
        'personas.lead': 'À qui vous parlez, et l’approche qui a une chance de le faire changer d’avis.',
        'personas.loading': 'Chargement des profils…',
        'personas.archetypesTitle': 'Archétypes',
        'personas.archetypesLeadHtml': 'Chaque archétype associe un <strong>système de valeurs</strong> à l’<strong>approche de persuasion</strong> la plus susceptible de le faire bouger. Sur chaque idée reçue, vous pouvez filtrer les réponses selon ces profils.',
        'personas.evsTitle': 'Dans la population française',
        'personas.evsLead': 'Part estimée de chaque profil parmi les Français.',
        'personas.methodLink': 'Comment ces profils ont été construits',
        'personas.methodLoading': 'Chargement…',
        'personas.methodError': 'Impossible de charger cette explication. Veuillez réessayer plus tard.',
        'personas.empty': 'Aucun profil disponible pour l’instant.',
        'personas.loadError': 'Impossible de charger les profils. Veuillez réessayer plus tard.',
        'personas.fallbackName': 'Profil',
        'personas.valueSystem': 'Système de valeurs',
        'personas.engagement': 'Engagement : {value}',
        'personas.howToReach': 'Comment lui parler :',
        'personas.viewProfile': 'Voir comment lui parler →',

        // Persona detail
        'persona.metaTitle': 'Profil - re.inform',
        'persona.metaDescription': 'Un profil : ses valeurs, et l’approche la plus susceptible de le faire changer d’avis.',
        'persona.back': '← Tous les profils',
        'persona.loading': 'Chargement du profil…',
        'persona.otherProfiles': 'Autres profils',
        'persona.none': 'Aucun profil sélectionné.',
        'persona.notFound': 'Profil introuvable.',
        'persona.loadError': 'Impossible de charger ce profil. Veuillez réessayer plus tard.',
        'persona.share': 'Environ {pct} de la population française.',
        'persona.defines': 'Ce qui le définit',
        'persona.messengers': 'Messagers de confiance :',
        'persona.howToChange': 'Comment le faire changer d’avis',
        'persona.do': 'À faire',
        'persona.avoid': 'À éviter',
        'persona.whoToBring': 'Qui faire intervenir',
        'persona.claimsTitle': 'Ce que vous entendrez',
        'persona.claimsEmpty': 'Aucune idée reçue n’est encore liée à ce profil.',
        'persona.claimsAll': { one: 'Voir l’idée reçue →', other: 'Voir les {n} idées reçues →' },
        'persona.claimsLead': 'Les idées reçues qui visent ce profil, les plus urgentes d’abord.',
        'persona.axes': 'Axes de valeurs les plus distinctifs',
        'dim.epistemic_style.name': 'Rapport au savoir',
        'dim.epistemic_style.left': 'Données & institutions',
        'dim.epistemic_style.right': 'Intuition & alternatives',
        'dim.moral_foundations.name': 'Fondements moraux',
        'dim.moral_foundations.left': 'Soin & justice',
        'dim.moral_foundations.right': 'Loyauté & tradition',
        'dim.threat_sensitivity.name': 'Rapport au changement',
        'dim.threat_sensitivity.left': 'Ouvert au changement',
        'dim.threat_sensitivity.right': 'Défense du statu quo',
        'persona.previous': 'Précédent',
        'persona.next': 'Suivant',
        'importance.title': { one: 'Importance : {avg}/5 sur {n} vote', other: 'Importance : {avg}/5 sur {n} votes' },
        'importance.unrated': 'Importance non évaluée',
        'importance.unratedTitle': 'Personne n’a encore dit si c’était urgent',
        'importance.contributions': { one: '{n} contribution', other: '{n} contributions' },
        'importance.contributionsTitle': 'Réponses, notes et ressources réunies',

        // ---- Claim page (argument.html) ----
        'claim.metaTitle': 'Idée reçue - re.inform',
        'claim.metaDescription': 'Les réponses à une idée reçue, adaptées à la personne à qui vous parlez.',
        'claim.back': '← Retour aux idées reçues',
        'claim.loading': 'Chargement de l’idée reçue…',
        'claim.label': 'Idée reçue',
        'claim.fallback': 'Idée reçue',
        'claim.none': 'Aucune idée reçue sélectionnée.',
        'claim.loadError': 'Impossible de charger cette idée reçue. Veuillez réessayer plus tard.',
        'claim.notFound': 'Idée reçue introuvable',
        'claim.notFoundHtml': 'Elle n’existe pas ou a été retirée. <a href="index.html">Voir toutes les idées reçues</a>',
        'claim.responses': 'Réponses',
        'claim.responsesCount': { one: '{n} réponse', other: '{n} réponses' },
        'claim.forWhom': 'Pour qui :',
        'claim.anyone': 'Tout le monde',
        'claim.seenWith': 'Réponses pour :',
        'claim.gapsTitle': 'Profils encore sans réponse adaptée',
        'claim.gapsHint': 'Vous savez comment toucher l’un d’eux ? Écrivez la première réponse : c’est ce qui aide le plus ici.',
        'claim.writeFor': 'Écrire pour {name}',
        'claim.emptyForProfile': 'Pas encore de réponse pour ce profil.',
        'claim.emptyAll': 'Pas encore de réponse à cette idée reçue.',
        'claim.writeFirst': 'Écrire la première',

        'response.general': 'Général',
        'response.ratingTitle': { one: '{avg}/5 sur {n} note', other: '{avg}/5 sur {n} notes' },
        'response.notRated': 'Pas encore notée',
        'response.noSource': 'Sans source',
        'response.check': 'Vérifier la source',
        'response.changeCheck': 'Modifier votre vérification',
        'response.readMore': 'Lire la suite',
        'response.readLess': 'Réduire',
        'response.copy': 'Copier',
        'response.rate': 'Noter',
        'response.myRating': 'Votre note : {n}/5',
        'response.improve': 'Améliorer',
        'response.copySources': 'Sources :',
        'response.copied': 'Copié ! Dites-nous si ça a marché dans une vraie conversation.',
        'response.copyFailed': 'Copie automatique impossible : sélectionnez le texte et copiez-le.',

        // Source-verified stamp and checks
        'stamp.verified': 'Source vérifiée',
        'stamp.verifiedTitle': 'Des vérificateurs ont confirmé que les sources citées appuient cette réponse.',
        'stamp.needsWork': 'Source à revoir',
        'stamp.needsWorkTitle': 'Des vérificateurs ont trouvé que les sources citées n’appuient pas entièrement cette réponse.',
        'stamp.pending': 'Source pas encore vérifiée',
        'stamp.pendingTitle': 'Aucun vérificateur n’a encore confirmé ces sources.',
        'stamp.inProgress': 'Vérification en cours ({n})',
        'verdict.supports': 'Oui, elles la confirment',
        'verdict.partial': 'En partie seulement',
        'verdict.doesNotSupport': 'Non',
        'verdict.broken': 'Lien cassé',
        'check.prompt': 'Ouvrez les sources ci-dessus. Confirment-elles cette réponse ?',
        'check.verdictLabel': 'Votre verdict',
        'check.noteLabel': 'Note pour l’auteur (facultatif)',
        'check.notePlaceholder': 'Facultatif : qu’est-ce qui manque ou cloche ?',
        'check.submit': 'Envoyer la vérification',
        'check.hint': 'Votre vérification compte pour le tampon. Les points arrivent une fois le verdict établi.',
        'check.error': 'Impossible d’enregistrer votre vérification.',
        'check.points': ' +{n} de réputation.',
        'check.verified': 'Source vérifiée.{points}',
        'check.needsWork': 'Signalé : les sources sont à revoir.{points}',
        'check.saved': 'Vérification enregistrée. Encore une et ce sera tranché.',
        'check.networkError': 'Problème de connexion pendant l’enregistrement de votre vérification.',

        // Rating a response
        'rate.prompt': 'Est-ce que ça marcherait ?',
        'rate.star': { one: '{n} étoile', other: '{n} étoiles' },
        'rate.thanks': 'Merci, noté {n} ★',
        'rate.updated': 'Note modifiée : {n} ★',
        'rate.promptChange': 'Votre note : {n} ★. Cliquez pour la changer.',
        'rate.saved': 'Note enregistrée. Vous l’avez déjà essayée pour de vrai ?',
        'rate.error': 'Impossible d’enregistrer la note.',
        'rate.networkError': 'Problème de connexion pendant l’enregistrement de la note.',

        // Response composer
        'composer.title': 'Écrire une réponse',
        'composer.titleFor': 'Écrire une réponse pour {name}',
        'composer.titleImprove': 'Proposer une version améliorée',
        'composer.hint': 'Vous répondez à {name}, {label}.',
        'composer.hintApproach': 'Piste :',
        'composer.persona': 'À qui répondez-vous ?',
        'composer.personaHint': 'Chaque réponse s’adresse à un seul profil.',
        'composer.choosePersona': 'Choisissez un profil…',
        'composer.text': 'Votre réponse',
        'composer.textPlaceholder': 'Écrivez une réponse qui parlerait à cette personne…',
        'composer.sources': 'Sources (preuves)',
        'composer.sourcesHint': 'Ajoutez un ou plusieurs liens qui appuient cette réponse.',
        'composer.sourcePlaceholder': 'https://exemple.fr/preuve',
        'composer.addSource': '+ Ajouter une autre source',
        'composer.submit': 'Publier',
        'composer.sourceInvalid': 'Les liens doivent commencer par http:// ou https://',
        'composer.publishing': 'Publication…',
        'composer.loginToast': 'Pour écrire une réponse, il vous faut un compte gratuit.',
        'composer.sessionExpiredHtml': 'Session expirée, <a href="{loginHref}">reconnectez-vous</a>. Votre texte est gardé.',
        'composer.sessionExpired': 'Session expirée, reconnectez-vous. Votre texte est gardé.',
        'composer.draftSaved': 'Brouillon enregistré',
        'composer.draftError': 'Brouillon non enregistré.',
        'composer.draftRestored': 'Votre brouillon est revenu. Vous pouvez le publier.',
        'composer.needText': 'Écrivez une réponse.',
        'composer.needPersona': 'Choisissez à qui s’adresse cette réponse.',
        'composer.published': 'Réponse publiée, merci pour votre contribution !',
        'composer.error': 'Impossible de publier la réponse.',

        // ---- Topics (tags in the data are English codes) ----
        'topic.climate': 'Climat',
        'topic.society': 'Société',
        'topic.policy': 'Politiques publiques',
        'topic.energy': 'Énergie',
        'topic.transport': 'Transports',
        'topic.economy': 'Économie',
        'topic.agriculture': 'Agriculture',
        'topic.media': 'Médias',
        'topic.technology': 'Technologie',
        'topic.biodiversity': 'Biodiversité',
        'topic.adaptation': 'Adaptation',
        'topic.environment': 'Environnement',
        'topic.communication': 'Communication',
        'topic.france': 'France',
        'topic.solutions': 'Solutions',
        'topic.tools': 'Outils',
        'topic.values': 'Valeurs',
        'topic.housing': 'Logement',
        'topic.persuasion': 'Persuasion',
        'topic.dialogue': 'Dialogue',
        'topic.polarization': 'Polarisation',
        'topic.matieres-premieres': 'Matières premières',
        'topic.pollution': 'Pollution',
        'topic.ecology': 'Écologie',
        'topic.politics': 'Politique',

        // ---- Persuasion formats (codes in the persona data) ----
        'format.calm_discussion': 'Discussion calme',
        'format.calm_discussion_evidence': 'Discussion calme et données',
        'format.moral_care_fairness': 'Soin et justice',
        'format.moral_tradition_order': 'Tradition et ordre',
        'format.example_social_proof': 'Exemple et preuve sociale',
        'format.authority_rules': 'Autorité et règles',

        // ---- Filter bar (library, articles) ----
        'filter.sort': 'Trier',
        'filter.search': 'Rechercher…',
        'filter.nothing': 'Rien à filtrer pour l’instant.',
        'filter.noMatch': 'Aucun résultat',
        'filter.selectAll': 'Tout sélectionner',
        'filter.searchTopics': 'Rechercher un thème…',
        'filter.selectedCount': { one: '{n} sélectionné', other: '{n} sélectionnés' },
        'filter.clearAll': 'Tout effacer',
        'filter.chipLabel': '{label} :',
        'filter.topic': 'Thème',
        'filter.type': 'Type',
        'filter.publisher': 'Éditeur',
        'filter.topicsCount': '{n} thèmes',
        'filter.allTopics': 'Tous les thèmes',

        // ---- Library ----
        'library.metaTitle': 'Bibliothèque - re.inform',
        'library.metaDescription': 'Toutes les ressources externes vérifiées (articles, études, infographies, vidéos, données), à filtrer par type, thème et éditeur.',
        'library.eyebrow': 'Vérifiées par notre équipe',
        'library.title': 'Bibliothèque',
        'library.lead': 'Toutes les sources externes réunies par la communauté et vérifiées par un administrateur : articles, études, infographies, vidéos et données. Parcourez-les librement, quelle que soit l’idée reçue à laquelle elles ont d’abord servi à répondre.',
        'library.statsLabel': 'Taille de la bibliothèque',
        'library.loading': 'Chargement des ressources…',
        'library.empty': 'Aucune ressource ne correspond à ces filtres pour l’instant.',
        'library.loadError': 'Impossible de charger la bibliothèque. Veuillez réessayer.',
        'library.sortCited': 'Les plus citées',
        'library.sortRecent': 'Les plus récentes',
        'library.sortTitle': 'A–Z',
        'library.search': 'Rechercher une ressource…',
        'library.statShown': { one: 'ressource affichée', other: 'ressources affichées' },
        'library.statPublishers': { one: 'éditeur', other: 'éditeurs' },
        'library.statCited': { one: 'citée dans des réponses', other: 'citées dans des réponses' },
        'library.cited': { one: 'Citée dans {n} réponse', other: 'Citée dans {n} réponses' },
        'library.sharedBy': 'Partagée par {name}',
        'library.context': 'Contexte ↗',
        'libraryType.article': 'Article',
        'libraryType.study': 'Étude / publication',
        'libraryType.report': 'Rapport / PDF',
        'libraryType.infographic': 'Infographie',
        'libraryType.video': 'Vidéo',
        'libraryType.podcast': 'Podcast / audio',
        'libraryType.dataset': 'Données',
        'libraryType.website': 'Site web',

        // ---- Articles (resources.html) ----
        'articles.metaTitle': 'Articles - re.inform',
        'articles.metaDescription': 'Une sélection d’articles et de ressources, classés selon leur efficacité.',
        'articles.title': 'Articles',
        'articles.lead': 'Une sélection d’articles et de ressources, classés selon leur efficacité.',
        'articles.loading': 'Chargement des articles…',
        'articles.curatedTitle': 'Notre sélection',
        'articles.curatedLead': 'Nos ressources les plus efficaces, appuyées par la recherche et classées selon leur efficacité.',
        'articles.empty': 'Aucun article sélectionné pour l’instant.',
        'articles.loadError': 'Impossible de charger les articles. Veuillez réessayer plus tard.',
        'articles.sortRating': 'Efficacité',
        'articles.sortRecent': 'Les plus récents',
        'articles.sortTitle': 'A–Z',
        'articles.search': 'Rechercher un article…',
        'articles.read': 'Lire l’article',
        'articles.anonymous': 'Auteur inconnu',

        // ---- Article reader ----
        'article.metaTitle': 'Article - re.inform',
        'article.metaDescription': 'Lire un article re.inform.',
        'article.back': '← Retour aux articles',
        'article.backMethodology': '← Retour à la méthodologie',
        'article.title': 'Article',
        'article.loading': 'Chargement de l’article…',
        'article.invalid': 'Source de l’article manquante ou invalide.',
        'article.sourceHtml': 'Source : <a href="{href}" target="_blank" rel="noopener noreferrer">texte d’origine (Markdown)</a>',
        'article.loadError': 'Impossible de charger l’article. Veuillez réessayer plus tard.',

        // ---- Admin ----
        'admin.metaTitle': 'Administration - re.inform',
        'admin.metaDescription': 'Espace d’administration pour relire les contributions de la communauté.',
        'admin.loginTitle': 'Connexion administrateur',
        'admin.password': 'Mot de passe :',
        'admin.passwordPlaceholder': 'Mot de passe administrateur',
        'admin.invalidPassword': 'Mot de passe incorrect',
        'admin.submissionsTitle': 'Relire les contributions',
        'admin.logout': 'Se déconnecter',
        'admin.submissionsLoading': 'Chargement des contributions…',
        'admin.submissionsEmpty': 'Aucune contribution pour l’instant.',
        'admin.submissionsLoadError': 'Impossible de charger les contributions.',
        'admin.resourcesTitle': 'Relire les ressources',
        'admin.show': 'Afficher :',
        'admin.filterPending': 'À relire',
        'admin.filterApproved': 'Validées',
        'admin.filterRejected': 'Refusées',
        'admin.filterAll': 'Toutes',
        'admin.approve': 'Valider',
        'admin.reject': 'Refuser',
        'admin.fieldTags': 'Thèmes (séparés par des virgules) :',
        'admin.fieldTagsPlaceholder': 'ex. : immigration, économie, climat',
        'admin.confirmApproval': 'Confirmer la validation',
        'admin.submissionNumber': 'Contribution n° {id}',
        'admin.statusPending': 'En attente',
        'admin.statusApproved': 'Validée',
        'admin.statusRejected': 'Refusée',
        'admin.claim': 'Idée reçue :',
        'admin.response': 'Réponse :',
        'admin.source': 'Source :',
        'admin.submitted': 'Envoyée le :',
        'admin.author': 'Auteur :',
        'admin.newClaim': 'nouvelle idée reçue',
        'admin.notProvided': 'Non renseignée',
        'admin.confirmReject': 'Voulez-vous vraiment refuser cette contribution ?',
        'admin.approved': 'Contribution validée.',
        'admin.approveFailed': 'Échec de la validation : {error}',
        'admin.rejected': 'Contribution refusée.',
        'admin.rejectFailed': 'Échec du refus : {error}',
        'admin.unknownError': 'erreur inconnue',
        'admin.reviewLoading': 'Chargement des ressources…',
        'admin.reviewCounts': '{pending} à relire · {approved} validées · {rejected} refusées',
        'admin.reviewLoadError': 'Impossible de charger les ressources à relire.',
        'admin.reviewEmpty': 'Rien ici : la liste est vide.',
        'admin.byProvenance': 'validée automatiquement d’après sa provenance ({origins})',
        'admin.reviewedBy': 'relue par {names}',
        'admin.neverReviewed': 'jamais relue',
        'admin.reviewTitle': 'Titre',
        'admin.reviewType': 'Type',
        'admin.reviewTopics': 'Thèmes (séparés par des virgules)',
        'admin.reviewDescription': 'Description',
        'admin.reviewCited': { one: 'Citée dans {n} réponse · {provenance}', other: 'Citée dans {n} réponses · {provenance}' },
        'admin.reviewError': 'Impossible d’enregistrer la relecture.',
        'admin.reviewApproved': 'Ressource validée.',
        'admin.reviewRejected': 'Ressource refusée.',

        // ---- API error codes ({error, code} responses; see backend/README.md) ----
        'error.auth_required': 'Connectez-vous pour continuer.',
        'error.forbidden_role': 'Vous n’avez pas les droits pour faire cela.',
        'error.forbidden': 'Vous n’avez pas les droits pour faire cela.',
        'error.not_your_account': 'Ce compte n’est pas le vôtre.',
        'error.fields_required': 'Tous les champs sont obligatoires.',
        'error.credentials_required': 'Saisissez votre nom d’utilisateur et votre mot de passe.',
        'error.invalid_credentials': 'Nom d’utilisateur ou mot de passe incorrect.',
        'error.username_too_short': 'Le nom d’utilisateur doit faire au moins 3 caractères.',
        'error.username_too_long': 'Le nom d’utilisateur doit faire au plus 30 caractères.',
        'error.username_invalid_chars': 'Le nom d’utilisateur ne peut contenir que des lettres, des chiffres, des points, des tirets et des tirets bas.',
        'error.password_too_short': 'Le mot de passe doit faire au moins 6 caractères.',
        'error.account_exists': 'Ce nom d’utilisateur ou cette adresse e-mail est déjà utilisé.',
        'error.user_not_found': 'Membre introuvable.',
        'error.admin_auth_required': 'Session administrateur expirée : reconnectez-vous.',
        'error.invalid_admin_password': 'Mot de passe incorrect',
        'error.admin_password_not_set': 'Le mot de passe administrateur n’est pas configuré sur le serveur.',
        'error.submission_not_pending': 'Cette contribution n’est plus en attente de relecture.',
        'error.invalid_vote_type': 'Type de vote inconnu.',
        'error.claim_not_found': 'Idée reçue introuvable.',
        'error.answer_not_found': 'Réponse introuvable.',
        'error.claim_text_required': 'Écrivez l’idée reçue.',
        'error.answer_text_required': 'Écrivez une réponse.',
        'error.invalid_argument_id': 'Idée reçue invalide.',
        'error.invalid_counter_argument_id': 'Réponse invalide.',
        'error.persona_required': 'Choisissez à qui s’adresse cette réponse.',
        'error.persona_not_found': 'Profil introuvable.',
        'error.persona_not_named': 'Choisissez un profil dans la liste.',
        'error.invalid_persona_id': 'Profil invalide.',
        'error.tags_not_list': 'Les thèmes sont mal formés.',
        'error.endpoint_moved': 'Cette fonction a changé de place : rechargez la page.',
        'error.invalid_score': 'Note invalide.',
        'error.anon_vote_limit': 'Créez un compte gratuit pour continuer à voter.',
        'error.anon_votes_ip_limit': 'Trop de votes depuis ce réseau aujourd’hui. Créez un compte gratuit pour continuer à voter.',
        'error.draft_text_too_long': 'Ce brouillon est trop long pour être enregistré.',
        'error.draft_sources_too_long': 'Trop de sources dans ce brouillon, ou une source trop longue.',
        'error.invalid_sources': 'Les sources sont mal formées.',
        'error.drafts_limit': 'Vous avez trop de brouillons : supprimez-en un pour enregistrer celui-ci.',
        'error.draft_not_found': 'Ce brouillon n’existe plus.',
        'error.invalid_url': 'Les liens doivent commencer par http:// ou https://',
        'error.url_required': 'Ajoutez un lien.',
        'error.topics_not_list': 'Les thèmes sont mal formés.',
        'error.invalid_status': 'Statut invalide.',
        'error.invalid_type': 'Type invalide.',
        'error.invalid_verdict': 'Verdict invalide.',
        'error.check_login_required': 'Connectez-vous pour vérifier une source.',
        'error.check_sourcer_role_required': 'Prenez le rôle de vérificateur pour vérifier les sources.',
        'error.check_own_debunk': 'Vous ne pouvez pas vérifier vos propres réponses.',
        'error.check_no_sources': 'Cette réponse n’a pas de source à vérifier.',
        'error.avatar_not_found': 'Pas de photo de profil.',
        'error.file_required': 'Aucun fichier envoyé.',
        'error.unsupported_image': 'Format d’image non pris en charge (JPG, PNG, GIF ou WebP).',
        'error.image_too_large': 'Image trop lourde (3 Mo maximum).',
        'error.tag_required': 'Choisissez un thème.',
        'error.already_following_tag': 'Vous suivez déjà ce thème.',
        'error.not_following_tag': 'Vous ne suivez pas ce thème.',
        'error.follow_self': 'Vous ne pouvez pas vous suivre vous-même.',
        'error.already_following_user': 'Vous suivez déjà cette personne.',
        'error.not_following_user': 'Vous ne suivez pas cette personne.',
        'error.endorse_self': 'Vous ne pouvez pas vous soutenir vous-même.',
        'error.already_endorsed': 'Vous soutenez déjà cette personne.',
        'error.reason_required': 'Indiquez une raison.',
        'error.nomination_fields_required': 'Indiquez un membre et une raison.',
        'error.unknown_role': 'Rôle inconnu.',
        'error.role_already_held': 'Vous avez déjà ce rôle.',
        'error.role_not_held': 'Vous n’avez pas ce rôle.',
        'error.rate_limited': 'Limite atteinte pour aujourd’hui. Réessayez demain.',
        'error.invalid_input': 'Certaines informations sont invalides.',
        'error.not_found': 'Introuvable.',
        'error.method_not_allowed': 'Action impossible.',
        'error.payload_too_large': 'Envoi trop volumineux.',
        'error.bad_request': 'Requête invalide.',
        'error.server_error': 'Erreur du serveur. Veuillez réessayer.',
    },
};

// Fallback for API errors without a `code` (older endpoints): matched on the
// exact English text. Codes (`error.<code>` above) always win. See apiError().
const API_ERRORS = {
    fr: {
        'Invalid username or password': 'Nom d’utilisateur ou mot de passe incorrect.',
        'Username and password required': 'Saisissez votre nom d’utilisateur et votre mot de passe.',
        'All fields are required': 'Tous les champs sont obligatoires.',
        'Username must be at least 3 characters': 'Le nom d’utilisateur doit faire au moins 3 caractères.',
        'Password must be at least 6 characters': 'Le mot de passe doit faire au moins 6 caractères.',
        'Username or email already exists': 'Ce nom d’utilisateur ou cette adresse e-mail est déjà utilisé.',
        'Invalid password': 'Mot de passe incorrect',
        'Create a free account to keep voting.': 'Créez un compte gratuit pour continuer à voter.',
        'Too many votes from this network today. Create a free account to keep voting.': 'Trop de votes depuis ce réseau aujourd’hui. Créez un compte gratuit pour continuer à voter.',
        'Rate limit exceeded. Maximum 50 votes per day.': 'Limite atteinte : 50 votes par jour au maximum.',
        'Rate limit exceeded. Maximum 50 source checks per day.': 'Limite atteinte : 50 vérifications par jour au maximum.',
        'Image is too large (max 3 MB)': 'Image trop lourde (3 Mo maximum).',
        'Unsupported image format (use JPG, PNG, GIF or WebP)': 'Format d’image non pris en charge (JPG, PNG, GIF ou WebP).',
        'No file provided': 'Aucun fichier envoyé.',
        'The URL must start with http:// or https://': 'Le lien doit commencer par http:// ou https://',
        'Links must be full web addresses starting with http:// or https://': 'Les liens doivent commencer par http:// ou https://',
        'Source: Links must be full web addresses starting with http:// or https://': 'Les liens doivent commencer par http:// ou https://',
        'A resource URL is required': 'Ajoutez un lien.',
        'A target persona is required': 'Choisissez à qui s’adresse cette réponse.',
        'Choose a named persona': 'Choisissez un profil dans la liste.',
        'You already hold this role': 'Vous avez déjà ce rôle.',
        'You don’t hold this role': 'Vous n’avez pas ce rôle.',
        'Unknown role': 'Rôle inconnu.',
        'Argument not found': 'Idée reçue introuvable.',
        'Counter-argument not found': 'Réponse introuvable.',
        'User not found': 'Membre introuvable.',
        'Unauthorized': 'Vous devez être connecté.',
        'Insufficient permissions': 'Vous n’avez pas les droits pour faire cela.',
    },
};

// ---- Helpers ----

const __plural = new Intl.PluralRules(LOCALE);
const __number = new Intl.NumberFormat(LOCALE);

// French typography: non-breaking space before ? ! : ; » and after «.
function __typography(text) {
    if (LANG !== 'fr') return text;
    return text.replace(/ ([?!:;»])/g, ' $1').replace(/« /g, '« ');
}

function t(key, vars = {}) {
    const table = STRINGS[LANG] || {};
    let value = table[key];
    if (value === undefined) {
        console.warn(`[i18n] missing key: ${key}`);
        return key;
    }
    if (value && typeof value === 'object') {
        const n = Number(vars.n);
        value = value[__plural.select(Number.isFinite(n) ? n : 0)] ?? value.other;
    }
    const filled = String(value).replace(/\{(\w+)\}/g, (match, name) => {
        if (!(name in vars)) return match;
        const v = vars[name];
        return typeof v === 'number' ? __number.format(v) : String(v ?? '');
    });
    return __typography(filled);
}

// Catalog text when the key exists, otherwise `fallback` (e.g. a label the API
// sends for a value we have not translated yet).
function tOr(key, fallback, vars = {}) {
    const table = STRINGS[LANG] || {};
    return key in table ? t(key, vars) : (fallback ?? '');
}

// Interface text for an API error: by its `code` when we know it, else by the
// English message (older responses), else the caller's fallback key.
function apiError(data, fallbackKey) {
    const code = data && data.code;
    const table = STRINGS[LANG] || {};
    if (code && `error.${code}` in table) return t(`error.${code}`);
    const msg = data && data.error;
    const known = msg && (API_ERRORS[LANG] || {})[msg];
    return known ? __typography(known) : t(fallbackKey);
}

// True when the API says the visitor must log in (401, code auth_required).
function isAuthRequired(res, data) {
    return Boolean(res && res.status === 401 && (!data || !data.code || data.code === 'auth_required'));
}

// Numbers and dates in the interface language.
function formatNumber(value, decimals = null) {
    const opts = decimals == null ? {} : { minimumFractionDigits: decimals, maximumFractionDigits: decimals };
    return new Intl.NumberFormat(LOCALE, opts).format(Number(value) || 0);
}

function formatPercent(share, decimals = 1) {
    return new Intl.NumberFormat(LOCALE, {
        style: 'percent', minimumFractionDigits: decimals, maximumFractionDigits: decimals,
    }).format(Number(share) || 0);
}

function formatDate(value) {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleDateString(LOCALE, { day: 'numeric', month: 'long', year: 'numeric' });
}

function formatDateTime(value) {
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleString(LOCALE, { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// "il y a 3 jours", "hier"…
function formatTimeAgo(value) {
    const then = new Date(value).getTime();
    if (!then) return '';
    const secs = Math.max(1, Math.floor((Date.now() - then) / 1000));
    const units = [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600], ['minute', 60]];
    const rtf = new Intl.RelativeTimeFormat(LOCALE, { numeric: 'auto' });
    for (const [unit, s] of units) {
        const v = Math.floor(secs / s);
        if (v >= 1) return rtf.format(-v, unit);
    }
    return t('time.justNow');
}

// Fill every element marked with data-i18n* under `root` from the catalog.
function applyI18n(root = document) {
    root.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
    root.querySelectorAll('[data-i18n-html]').forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
    [['placeholder', 'i18nPlaceholder'], ['title', 'i18nTitle'], ['aria-label', 'i18nAriaLabel'], ['content', 'i18nContent']]
        .forEach(([attr, prop]) => {
            const sel = `[data-i18n-${attr}]`;
            root.querySelectorAll(sel).forEach((el) => el.setAttribute(attr, t(el.dataset[prop])));
        });
}

if (typeof document !== 'undefined') {
    document.documentElement.lang = LANG;
    document.addEventListener('DOMContentLoaded', () => applyI18n());
}
