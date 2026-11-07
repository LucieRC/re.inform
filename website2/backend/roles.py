"""
Contributor roles and the source-verified stamp.

Roles describe what a member *does* (function), separate from users.role,
which describes what they're *allowed* to do (rights). A member declares a
role and starts as an apprentice; demonstrated work promotes them to
confirmed. Tiers are never lowered automatically.

The first role mechanic is the source-verified stamp on a debunk: Sourcers
check whether the cited sources actually back it up, and the stamp is granted
once weighted checks agree. See docs/design/GAMIFICATION_DESIGN.md (Roles & stamps).

Pure logic over the argumentation store + database; no Flask in here.
"""
from typing import Any, Dict, List, Optional, Tuple

import argumentation_store as argstore
import database as db


# `icon` is a name from ICON_PATHS in frontend/js/main.js.
ROLE_CATALOG = {
    'wordsmith': {
        'name': 'Wordsmith',
        'icon': 'pen',
        'duty': 'Write debunks, and rewrite good ones for another persona.',
        'power': 'Suggest improved versions of other people’s debunks.',
        'promotion': 'Confirmed after 3 of your debunks are rated 3.5★ or more by at least 3 other members.',
        'public': True,
    },
    'sourcer': {
        'name': 'Sourcer',
        'icon': 'search',
        'duty': 'Check that the sources cited by a debunk really support it.',
        'power': 'Grant the “source verified” stamp. A confirmed Sourcer’s check counts double.',
        'promotion': 'Confirmed after 5 of your checks match the final verdict.',
        'public': True,
    },
    'field_agent': {
        'name': 'Field Agent',
        'icon': 'flask',
        'duty': 'Try debunks in real conversations and report what happened.',
        'power': 'Field reports are coming soon; you’ll be first to get field-test requests.',
        'promotion': None,
        # Saying you regularly argue with someone is personal; never shown publicly.
        'public': False,
    },
}

TIERS = ('apprentice', 'confirmed')

SOURCER_CONFIRM_AFTER = 5
WORDSMITH_CONFIRM_AFTER = 3
WORDSMITH_MIN_RATINGS = 3
WORDSMITH_MIN_AVG = 3.5

# A stamp needs this much weight on one side: one confirmed Sourcer (or a
# moderator), or two apprentices who agree.
STAMP_THRESHOLD = 2
WEIGHT_CONFIRMED = 2
WEIGHT_APPRENTICE = 1

# Points vest on resolution, never on submission.
POINTS_CHECK_AGREED = 5
POINTS_AUTHOR_VERIFIED = 5

POSITIVE_VERDICTS = {'supports'}


# ----------------------------------------
# Source checks → stamp status
# ----------------------------------------

def has_sources(ca: Dict[str, Any]) -> bool:
    return bool(ca.get('resource_ids')) or bool(ca.get('source_url'))


def check_weights() -> Dict[int, int]:
    """user_id → weight of their source checks. Moderators/admins count as confirmed."""
    weights = {
        uid: (WEIGHT_CONFIRMED if tier == 'confirmed' else WEIGHT_APPRENTICE)
        for uid, tier in db.get_role_tiers('sourcer').items()
    }
    for uid in db.get_user_ids_by_site_role('moderator', 'admin'):
        weights[uid] = WEIGHT_CONFIRMED
    return weights


def source_status(checks: List[Dict[str, Any]], weights: Dict[int, int]) -> Dict[str, Any]:
    """Resolve a debunk's checks into verified / needs_work / pending.

    Ties go to needs_work: the stamp is a promise to readers, so doubt wins.
    Checks from members who no longer hold the role keep apprentice weight.
    """
    support = against = 0
    for sc in checks:
        w = weights.get(sc.get('user_id'), WEIGHT_APPRENTICE)
        if sc.get('verdict') in POSITIVE_VERDICTS:
            support += w
        else:
            against += w

    if support >= STAMP_THRESHOLD and support > against:
        status = 'verified'
    elif against >= STAMP_THRESHOLD and against >= support:
        status = 'needs_work'
    else:
        status = 'pending'
    return {'status': status, 'support': support, 'against': against, 'count': len(checks)}


def _agrees(verdict: str, status: str) -> bool:
    if status == 'verified':
        return verdict in POSITIVE_VERDICTS
    if status == 'needs_work':
        return verdict not in POSITIVE_VERDICTS
    return False


def settle_source_rewards(store: Dict[str, Any], ca: Dict[str, Any], weights: Dict[int, int]) -> List[Tuple[int, int]]:
    """Pay out once a debunk's stamp resolves. Returns (user_id, points) pairs.

    Each check pays at most once (flagged `rewarded`), and the author is paid
    at most once for a verified source, so a stamp flipping back and forth
    can't be farmed. Mutates the store; the caller saves it.
    """
    checks = argstore.source_checks_by_counter(store).get(ca['id'], [])
    status = source_status(checks, weights)['status']
    if status == 'pending':
        return []

    payouts = []
    for sc in checks:
        if not sc.get('rewarded') and _agrees(sc['verdict'], status):
            sc['rewarded'] = True
            payouts.append((sc['user_id'], POINTS_CHECK_AGREED))

    author = ca.get('created_by_user_id')
    if status == 'verified' and isinstance(author, int) and not ca.get('source_reward_paid'):
        ca['source_reward_paid'] = True
        payouts.append((author, POINTS_AUTHOR_VERIFIED))
    return payouts


# Why a source check is refused: stable code → English text (the frontend maps
# the code to its own wording).
CHECK_REFUSALS = {
    'check_login_required': 'Log in to check sources.',
    'check_sourcer_role_required': 'Take the Sourcer role on your profile to check sources.',
    'check_own_debunk': 'You can’t check the sources of your own debunk.',
    'check_no_sources': 'This debunk has no source to check.',
}


def can_check_source(user_id: Optional[int], site_role: Optional[str], ca: Dict[str, Any],
                     sourcer_ids: set) -> Tuple[bool, Optional[str]]:
    """Whether this user may check this debunk's sources, and if not, the refusal code."""
    if user_id is None:
        return False, 'check_login_required'
    if user_id not in sourcer_ids and site_role not in ('moderator', 'admin'):
        return False, 'check_sourcer_role_required'
    if ca.get('created_by_user_id') == user_id:
        return False, 'check_own_debunk'
    if not has_sources(ca):
        return False, 'check_no_sources'
    return True, None


def source_check_queue(store: Dict[str, Any], user_id: int, weights: Dict[int, int],
                       limit: int = 3) -> List[Dict[str, Any]]:
    """Debunks waiting on a check this user can give: fewest checks, oldest first."""
    args_by_id, personas_by_id = argstore.build_indexes(store)
    res_by_id = argstore.resources_by_id(store)
    checks_by_ca = argstore.source_checks_by_counter(store)

    candidates = []
    for ca in argstore.published(store.get('counter_arguments')):
        if not isinstance(ca.get('id'), int):
            continue
        if not argstore.is_published(args_by_id.get(ca.get('argument_id'))):
            continue
        if ca.get('created_by_user_id') == user_id or not has_sources(ca):
            continue
        checks = checks_by_ca.get(ca['id'], [])
        if any(sc.get('user_id') == user_id for sc in checks):
            continue
        if source_status(checks, weights)['status'] != 'pending':
            continue
        candidates.append((len(checks), str(ca.get('created_at') or ''), ca))

    candidates.sort(key=lambda c: (c[0], c[1]))

    out = []
    for n_checks, _created, ca in candidates[:limit]:
        arg = args_by_id.get(ca.get('argument_id')) or {}
        persona = personas_by_id.get(ca.get('persona_archetype_id')) or {}
        resources = [
            {'url': r.get('url'), 'title': r.get('title')}
            for r in (res_by_id.get(rid) for rid in ca.get('resource_ids') or [])
            if r
        ] or [{'url': ca.get('source_url'), 'title': None}]
        text = str(ca.get('text') or '')
        out.append({
            'counter_argument_id': ca['id'],
            'argument_id': ca.get('argument_id'),
            'argument_text': arg.get('text'),
            'text': text[:200].rstrip() + '…' if len(text) > 200 else text,
            'persona': persona.get('name'),
            'resources': resources,
            'check_count': n_checks,
        })
    return out


# ----------------------------------------
# Role progress and promotion
# ----------------------------------------

def _sourcer_done(store: Dict[str, Any], user_id: int, weights: Dict[int, int]) -> int:
    """How many of this user's checks match the resolved verdict."""
    checks_by_ca = argstore.source_checks_by_counter(store)
    done = 0
    for checks in checks_by_ca.values():
        mine = next((sc for sc in checks if sc.get('user_id') == user_id), None)
        if mine and _agrees(mine['verdict'], source_status(checks, weights)['status']):
            done += 1
    return done


def _wordsmith_done(store: Dict[str, Any], user_id: int) -> int:
    """How many of this user's debunks are well rated by other members."""
    scores: Dict[int, List[int]] = {}
    for r in store.get('ratings', []) or []:
        if isinstance(r, dict) and isinstance(r.get('score'), int) and r.get('user_id') != user_id:
            scores.setdefault(r.get('counter_argument_id'), []).append(r['score'])

    done = 0
    for ca in argstore.published(store.get('counter_arguments')):
        if ca.get('created_by_user_id') != user_id:
            continue
        s = scores.get(ca.get('id'), [])
        if len(s) >= WORDSMITH_MIN_RATINGS and sum(s) / len(s) >= WORDSMITH_MIN_AVG:
            done += 1
    return done


def _progress(role: str, store: Dict[str, Any], user_id: int, weights: Dict[int, int]) -> Optional[Dict[str, Any]]:
    if role == 'sourcer':
        return {'done': _sourcer_done(store, user_id, weights), 'needed': SOURCER_CONFIRM_AFTER,
                'unit': 'checks matching the final verdict', 'unit_code': 'checks_matching_verdict'}
    if role == 'wordsmith':
        return {'done': _wordsmith_done(store, user_id), 'needed': WORDSMITH_CONFIRM_AFTER,
                'unit': 'debunks rated 3.5★+ by 3 members', 'unit_code': 'debunks_well_rated'}
    return None


def describe_roles(store: Dict[str, Any], user_id: int, *, public_only: bool = False,
                   weights: Optional[Dict[int, int]] = None) -> List[Dict[str, Any]]:
    """The user's roles with catalog info and progress, promoting where earned."""
    weights = weights if weights is not None else check_weights()
    out = []
    for held in db.get_user_roles(user_id):
        role = held['role']
        meta = ROLE_CATALOG.get(role)
        if not meta or held['scope_kind'] or (public_only and not meta['public']):
            continue

        progress = _progress(role, store, user_id, weights)
        tier = held['tier']
        if tier == 'apprentice' and progress and progress['done'] >= progress['needed']:
            tier = 'confirmed'
            db.set_user_role_tier(user_id, role, tier)

        entry = {
            'role': role,
            'name': meta['name'],
            'icon': meta['icon'],
            'tier': tier,
            'since': held['created_at'],
        }
        if not public_only:
            entry.update({'origin': held['origin'], 'progress': progress})
        out.append(entry)
    return out


def catalog() -> List[Dict[str, Any]]:
    return [{'role': key, **meta} for key, meta in ROLE_CATALOG.items()]
